"""
Author: Huynh Van Thong
Department of AI Convergence, Chonnam Natl. Univ.
"""
import gc
import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_io as tfio
import tensorflow_hub as hub
import librosa
import pandas as pd
import cv2
import numpy as np
import pathlib
import argparse

DEFAULT_SR = 16000

model_link_dict = {'efficientnet-b0': ("https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1", 224),
                   'resnetv2-50': ("https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/4", 224)}


def create_feature_extractor_model(modelname='efficientnet-b0'):
    target_size = model_link_dict[modelname][1]
    target_link = model_link_dict[modelname][0]

    model = tf.keras.Sequential([  # layers.experimental.preprocessing.Resizing(target_size, target_size),
        layers.experimental.preprocessing.Rescaling(1. / 255),
        hub.KerasLayer(target_link, trainable=False)])
    model.build([None, 224, 224, 3])
    return model


def resize_pad(img, target=224):
    h, w = img.shape[:2]
    if h < w:
        pad_h = w - h
        pad_w = 0
    else:
        pad_w = h - w
        pad_h = 0

    img = np.pad(img, ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2), (0, 0)),
                 mode='constant')
    if img.shape[0] != img.shape[1] or img.shape[2] != 3:
        print('Error in here, please stop ', img.shape)
        sys.exit(0)
    img = cv2.resize(img, (target, target))
    return img


def read_video(path, num_segments=-1, unlabelled_rows=None, get_audio=False):
    print(path)
    if not get_audio:
        cap = cv2.VideoCapture(path)
        vd_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(resize_pad(frame, 224), cv2.COLOR_BGR2RGB)
            vd_frames.append(frame)
        cap.release()

        vd_frames = np.array(vd_frames)
        if num_segments > 0:
            segment_len = int(vd_frames.shape[0] / num_segments)
            use_indexes = np.linspace(segment_len // 2, vd_frames.shape[0], num=num_segments, dtype=int, endpoint=False)
            vd_frames = vd_frames[use_indexes, :, :, :]
        # if unlabelled_rows is not None:
        #     vd_frames = vd_frames[np.logical_not(unlabelled_rows), :, :, :]

        # h, w = vd_frames.shape[1: 3]
        #
        #
        # vd_frames = np.pad(vd_frames, ((0, 0), (pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2), (0, 0)),
        #                    mode='constant')
        return vd_frames

    else:
        audio, sr = librosa.load(path)
        if sr != DEFAULT_SR:
            audio = librosa.resample(audio, sr, DEFAULT_SR)
        if num_segments > 0 and audio.shape[0] % num_segments > 0:
            num_pad = num_segments - (audio.shape[0] % num_segments)
            audio = np.pad(audio, ((0, num_pad)), mode='constant')

        audio = audio.reshape(num_segments, -1)
        # if unlabelled_rows is not None:
        #     audio = audio[np.logical_not(unlabelled_rows), :]

        return audio


def mediaeval_feature_extractor(split, visual=True):
    with open(os.path.join(dataset_root_path, '{}.txt'.format(split)), 'r') as fd:
        list_files = fd.readlines()

    list_files = [x.replace('\n', '') for x in list_files]
    model_created = {}
    prev_shape = {}

    if visual:
        use_model = ['resnetv2-50', 'efficientnet-b0']  # list(model_link_dict.keys())  #
        for model_id in use_model:
            model_created[model_id] = create_feature_extractor_model(model_id)
            prev_shape[model_id] = None
    else:
        module = tf.keras.Sequential([hub.KerasLayer('https://tfhub.dev/google/nonsemantic-speech-benchmark/trill-distilled/3',
                                                 arguments={'sample_rate': tf.constant(DEFAULT_SR, tf.int32)},
                                                 trainable=False, output_key='embedding',
                                                 output_shape=[None, 2048])])

    folder_write = pathlib.Path('{}/dataset/features_v2/{}'.format(dataset_root_path, split))
    folder_write.mkdir(parents=True, exist_ok=True)

    count = 0
    for vid in list_files:
        count += 1
        if count % 10 == 0 :
            print(count, '**', len(list_files))

        video_path = os.path.join(dataset_root_path, 'raw/{}/{}'.format(split, vid))
        video_anno_path = video_path.replace('.mp4', '_Valence-Arousal.txt')
        video_anno = pd.read_csv(video_anno_path, sep='\t').values

        feature_dict = {}

        if visual:
            vd_frames = read_video(video_path, num_segments=video_anno.shape[0])

            feature_dict.update({
                        'file_id': vid,
                        'timestamps': video_anno[:, 0].astype(np.int64),
                        'scores': video_anno[:, 1:]})

            for model_id in use_model:
                feature_dict['feature'] = model_created[model_id].predict(vd_frames, batch_size=128)
                np.save('{}/{}_{}.npy'.format(folder_write.__str__(), vid, model_id), feature_dict)
                _ = feature_dict.pop('feature')

            del vd_frames
        else:
            audio = read_video(video_path, num_segments=video_anno.shape[0], get_audio=True)
            # Audio embedding extraction
            # audio_emb = module(samples=audio, sample_rate=DEFAULT_SR)['embedding']
            audio_emb = module.predict(audio, batch_size=2048)

            # `emb` is a [batch_size, time, feature_dim] Tensor. In EvokedExpression, time=1
            # audio_emb.shape.assert_is_compatible_with([None, 512])
            audio_emb = np.squeeze(audio_emb)

            feature_dict['feature'] = audio_emb

            np.save('{}/{}_{}.npy'.format(folder_write.__str__(), vid, 'audio'), feature_dict)

            del audio_emb
            del feature_dict
            del audio

    print('Number of files: ', len(list_files))
    tf.keras.backend.clear_session()
    pass

def eev_feature_extractor(split, visual=True):
    data_csv = pd.read_csv('{}/eev/{}.csv'.format(dataset_root_path, split))
    id_header = 'Video ID' if split == 'test' else 'YouTube ID'
    video_ids = data_csv[id_header].unique()

    if not visual:
        module = tf.keras.Sequential([hub.KerasLayer('https://tfhub.dev/google/nonsemantic-speech-benchmark/trill-distilled/3',
                                                    arguments={'sample_rate': tf.constant(DEFAULT_SR, tf.int32)},
                                                    trainable=False, output_key='embedding',
                                                    output_shape=[None, 2048])])
    else:
        model_created = {}
        prev_shape = {}
        use_model = ['resnetv2-50', 'efficientnet-b0']  # list(model_link_dict.keys())  #
        for model_id in use_model:
            model_created[model_id] = create_feature_extractor_model(model_id)
            prev_shape[model_id] = None

    count = 0
    num_vids = len(video_ids)
    excluded_ids = np.loadtxt('excluded_files.txt', dtype=str)
    is_continue=True
    for vid in video_ids:
        gc.collect()
        if count % 10 == 0:
            print(count, "/", num_vids)
        count += 1

        if vid in excluded_ids:
            continue
        folder_write = pathlib.Path('{}/dataset/features_v2/{}'.format(dataset_root_path, split))
        folder_write.mkdir(parents=True, exist_ok=True)
        if os.path.isfile('{}/{}_{}.npy'.format(folder_write.__str__(), vid, 'audio')):
            tmp = np.load('{}/{}_{}.npy'.format(folder_write.__str__(), vid, 'audio'), allow_pickle=True)
            if tmp.item()['feature'].shape[-1] != 2048:
                print(tmp.item()['feature'].shape)
            else:
                continue

        feature_dict = {}
        current_frames = data_csv.loc[data_csv[id_header] == vid]
        if split in ['train', 'val']:
            scores = current_frames.values[:, 2:].astype(np.float32)
            unlabelled_rows = np.sum(scores, axis=-1) <= 1e-6
        else:
            scores = -1 * np.ones((current_frames.shape[0], 15)).astype(np.float32)
            unlabelled_rows = np.zeros(current_frames.shape[0], dtype=np.bool)

        unlabelled_rows = np.zeros(current_frames.shape[0], dtype=np.bool)
        if visual:
            try:
                vd_frames = read_video("{}/dataset/{}/{}.mp4".format(dataset_root_path, split, vid),
                                    num_segments=current_frames.shape[0], unlabelled_rows=unlabelled_rows)
                feature_dict.update({
                    'file_id': current_frames.values[:, 0][np.logical_not(unlabelled_rows)].astype(np.str),
                    'timestamps': current_frames.values[:, 1][np.logical_not(unlabelled_rows)].astype(np.int64),
                    'scores': scores[np.logical_not(unlabelled_rows), :]})

                for model_id in use_model:
                    feature_dict['feature'] = model_created[model_id].predict(vd_frames, batch_size=128)
                    np.save('{}/{}_{}.npy'.format(folder_write.__str__(), vid, model_id), feature_dict)
                    _ = feature_dict.pop('feature')
            except:
                with open('excluded_files_v3.txt', 'a') as fd:
                    fd.write('{} {}\n'.format(split, vid))
                break
                continue

            del vd_frames

        else:
            audio = read_video("{}/dataset/{}/{}.mp4".format(dataset_root_path, split, vid),
                            num_segments=current_frames.shape[0], unlabelled_rows=unlabelled_rows, get_audio=True)
            # Audio embedding extraction
            # audio_emb = module(samples=audio, sample_rate=DEFAULT_SR)['embedding']
            audio_emb = module.predict(audio, batch_size=2048)

            # `emb` is a [batch_size, time, feature_dim] Tensor. In EvokedExpression, time=1
            # audio_emb.shape.assert_is_compatible_with([None, 512])
            audio_emb = np.squeeze(audio_emb)

            feature_dict['feature'] = audio_emb

            np.save('{}/{}_{}.npy'.format(folder_write.__str__(), vid, 'audio'), feature_dict)

            del audio_emb
            del feature_dict
            del audio
        scores = None
        unlabelled_rows = None

    tf.keras.backend.clear_session()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Video feature extractor')
    parser.add_argument('--dataset', type=str, default='eev', help='Dataset eev or mediaeval18 (default: eev)')
    parser.add_argument('--dataset_root', type=str, default='/mnt/Work/Dataset/EEV/', help='Dataset root path (default: /mnt/Work/Dataset/EEV/)')
    parser.add_argument('--visual', action='store_true', help='Extract visual or audio (default: audio)')

    args = parser.parse_args()

    print(args.dataset, args.dataset_root, args.visual)

    dataset_root_path = args.dataset_root

    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    for split in ['train', 'val', 'test']:
        if args.dataset == 'eev':
            eev_feature_extractor(split, args.visual)
        elif args.dataset == 'mediaeval18':
            mediaeval_feature_extractor(split, args.visual)
        else:
            raise ValueError('Do not support {} dataset'.format(args.dataset))
