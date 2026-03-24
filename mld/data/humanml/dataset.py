from __future__ import annotations

import os
import random
import logging
import codecs as cs
from os.path import join as pjoin

import numpy as np
from rich.progress import track

import torch
from torch.utils.data import Dataset

from .scripts.motion_process import recover_from_ric
from .utils.word_vectorizer import WordVectorizer
from tqdm import tqdm
logger = logging.getLogger(__name__)


class MotionDataset(Dataset):
    def __init__(self, mean: np.ndarray, std: np.ndarray,
                 split_file: str, motion_dir: str, window_size: int,
                 tiny: bool = False, progress_bar: bool = True, **kwargs) -> None:
        self.data = []
        self.lengths = []
        id_list = []
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                id_list.append(line.strip())

        maxdata = 10 if tiny else 1e10
        if progress_bar:
            enumerator = enumerate(
                track(
                    id_list,
                    f"Loading HumanML3D {split_file.split('/')[-1].split('.')[0]}",
                ))
        else:
            enumerator = enumerate(id_list)

        count = 0
        for i, name in enumerator:
            if count > maxdata:
                break
            try:
                motion = np.load(pjoin(motion_dir, name + '.npy'))
                if motion.shape[0] < window_size:
                    continue
                self.lengths.append(motion.shape[0] - window_size)
                self.data.append(motion)
            except Exception as e:
                print(e)
                pass

        self.cumsum = np.cumsum([0] + self.lengths)
        if not tiny:
            logger.info("Total number of motions {}, snippets {}".format(len(self.data), self.cumsum[-1]))

        self.mean = mean
        self.std = std
        self.window_size = window_size

    def __len__(self) -> int:
        return self.cumsum[-1]

    def __getitem__(self, item: int) -> tuple:
        if item != 0:
            motion_id = np.searchsorted(self.cumsum, item) - 1
            idx = item - self.cumsum[motion_id] - 1
        else:
            motion_id = 0
            idx = 0
        motion = self.data[motion_id][idx:idx + self.window_size]
        "Z Normalization"
        motion = (motion - self.mean) / self.std
        return motion, self.window_size


class Text2MotionDataset(Dataset):

    def __init__(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        split_file: str,
        w_vectorizer: WordVectorizer,
        max_motion_length: int,
        min_motion_length: int,
        max_text_len: int,
        unit_length: int,
        motion_dir: str,
        text_dir: str,
        fps: int,
        padding_to_max: bool,
        njoints: int,
        tiny: bool = False,
        progress_bar: bool = True,
        lang_list: list | None = None,
        **kwargs,
    ) -> None:
        self.w_vectorizer = w_vectorizer
        self.max_motion_length = max_motion_length # 200
        self.min_motion_length = min_motion_length # 40
        self.max_text_len = max_text_len
        self.unit_length = unit_length # 4
        self.padding_to_max = padding_to_max # False for MLD, True for VAE
        self.njoints = njoints

        data_dict = {}
        id_list = []
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                id_list.append(line.strip())
        self.id_list = id_list
        count = 0
        bad_count = 0
        new_name_list = []
        length_list = []
        if lang_list is None:
            lang_list = ["en"]
        self.lang_list = lang_list
        logger.info("Text2MotionDataset languages: %s", lang_list)
        for lang in lang_list:
            for name in tqdm(id_list[:len(id_list)//1], desc=f'Loading {lang}-H3D '):
            #for name in tqdm(id_list, desc=f'Loading {lang}-H3D '):
                try:
                    motion = np.load(pjoin(motion_dir, name + ".npy"))
                    if (len(motion)) < self.min_motion_length or (len(motion) >=
                                                                  200):
                        bad_count += 1
                        continue
                    text_data = []
                    flag = False
                    with cs.open(pjoin(text_dir, name + ".txt")) as f:
                        for idx, line in enumerate(f.readlines()):
                            text_dict = {}
                            line_split = line.strip().split("#")
                            caption = line_split[0]
                            tokens = line_split[1].split(" ")
                            f_tag = float(line_split[2])
                            to_tag = float(line_split[3])
                            f_tag = 0.0 if np.isnan(f_tag) else f_tag
                            to_tag = 0.0 if np.isnan(to_tag) else to_tag
                            if lang == 'en':
                                text_dict['caption'] = caption
                            else:
                                with cs.open(pjoin(text_dir+'_'+lang, name + ".txt")) as fm:
                                    m_text = [line.replace('\n','') for line in fm.readlines()]
                                    text_dict['caption'] = m_text[idx]
                            text_dict["tokens"] = tokens

                            if f_tag == 0.0 and to_tag == 0.0:
                                flag = True
                                text_data.append(text_dict)
                            else:
                                try:
                                    n_motion = motion[int(f_tag * 20):int(to_tag *
                                                                          20)]
                                    if (len(n_motion)
                                        ) < self.min_motion_length or (
                                            (len(n_motion) >= 200)):
                                        continue
                                    new_name = (
                                        random.choice("ABCDEFGHIJKLMNOPQRSTUVW") +
                                        "_" + name)
                                    while new_name in data_dict:
                                        new_name = (random.choice(
                                            "ABCDEFGHIJKLMNOPQRSTUVW") + "_" +
                                                    name)
                                    new_name = lang + '_' + new_name
                                    data_dict[new_name] = {
                                        "motion": n_motion,
                                        "length": len(n_motion),
                                        "text": [text_dict],
                                    }
                                    new_name_list.append(new_name)
                                    length_list.append(len(n_motion))
                                except:
                                    # None
                                    print(line_split)
                                    print(line_split[2], line_split[3], f_tag,
                                          to_tag, name)
                                    # break

                    if flag:
                        data_dict[lang+'_'+name] = {
                            "motion": motion,
                            "length": len(motion),
                            "text": text_data,
                        }
                        new_name_list.append(lang+'_'+name)
                        length_list.append(len(motion))
                        count += 1
                
                except Exception as e:
                    print(e)
                    pass

        name_list, length_list = zip(
            *sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        if not tiny:
            logger.info(f"Reading {len(self.id_list)} motions from {split_file}.")
            logger.info(f"Total {len(name_list)} motions are used.")
            logger.info(f"{bad_count} motion sequences not within the length range of "
                        f"[{self.min_motion_length}, {self.max_motion_length}) are filtered out.")

        self.mean = mean
        self.std = std
        self.raw_mean = self.raw_std = None
        self.data_dict = data_dict
        self.name_list = name_list
        import pickle

        self.promt = None
        for p in (
            os.environ.get("DATASET_PROMPT_PKL"),
            "/data/wwj/AAAI26/whs/visualization/prompt.pkl",
        ):
            if p and os.path.isfile(p):
                with open(p, "rb") as f:
                    self.promt = pickle.load(f)
                break
    def __len__(self) -> int:
        return len(self.name_list)


    def __getitem__(self, idx: int) -> tuple:
        # print(self.name_list[idx])
        
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data["motion"], data["length"], data["text"]
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data["caption"], text_data["tokens"]
        # if 'en' in self.name_list[idx]:
        #     caption = self.promt['en'][self.name_list[idx].split('_')[-1]]
        # elif 'zh' in self.name_list[idx]:
        #     caption = self.promt['cn'][self.name_list[idx].split('_')[-1]]
        if len(tokens) < self.max_text_len:
            # pad with "unk"
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
            tokens = tokens + ["unk/OTHER"] * (self.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.max_text_len]
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        # Crop the motions in to times of 4, and introduce small variations
        if self.unit_length < 10:
            coin2 = np.random.choice(["single", "single", "double"])
        else:
            coin2 = "single"

        if coin2 == "double":
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        elif coin2 == "single":
            m_length = (m_length // self.unit_length) * self.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx + m_length]

        hint, hint_mask = None, None
        "Z Normalization"
        motion = (motion - self.mean) / self.std

        if self.padding_to_max:
            padding = np.zeros((self.max_motion_length - m_length, motion.shape[1]))
            motion = np.concatenate([motion, padding], axis=0)

        return (word_embeddings,
                pos_one_hots,
                caption,
                sent_len,
                motion,
                m_length,
                "_".join(tokens),
                self.name_list[idx])
