from random import randint
import numpy as np
import os
from PIL import Image
from PIL import ImageOps
from skimage.transform import resize
from scipy.ndimage import rotate, shift
from NTM_tensorflow.utils import one_hot_encode, baseN


class OmniglotDataLoader:
    def __init__(self, args):
        self.data = []
        self.image_size = (args.image_width, args.image_height)
        self.max_angle = args.max_angle
        self.max_shift = args.max_shift
        for dirname, subdirname, filelist in os.walk(args.data_dir):
            if filelist:
                self.data.append([Image.open(dirname + '/' + filename).copy() for filename in filelist])

        self.train_data = self.data[:args.n_train_classes]
        self.test_data = self.data[-args.n_test_classes:]

    def noisify_labels(self, labels, n_classes, noise_size=0.3, noise_strategy="random"):
        update_sample_size = int(len(labels) / n_classes) - 1
        assert update_sample_size * (1 - noise_size) >= 1, \
            "More than 1 sample per class remains unpolluted is required."
        total_update_samples = update_sample_size * n_classes
        num_noise_samples = int(total_update_samples * noise_size)
        num_clean_samples = total_update_samples - num_noise_samples
        clean_ids = np.arange(n_classes) * update_sample_size + np.random.randint(0, update_sample_size, n_classes)
        exclude = np.delete(np.arange(total_update_samples), clean_ids)
        more_ids = np.random.choice(exclude, num_clean_samples - n_classes, replace=False)
        clean_ids = np.concatenate([clean_ids, more_ids])
        noise_ids = np.delete(np.arange(total_update_samples), clean_ids)
        map_noise_ids = noise_ids // update_sample_size + noise_ids
        noise_samples = labels[map_noise_ids]
        if noise_strategy == "random":
            noise = np.random.randint(0, n_classes, len(map_noise_ids))
            while np.any(noise_samples == noise):
                noise = np.random.randint(0, n_classes, len(map_noise_ids))
            labels[map_noise_ids] = noise
        elif noise_strategy == "uniform":
            noise = np.random.permutation(noise_samples)
            while np.any(noise_samples == noise):
                noise = np.random.permutation(noise_samples)
            labels[map_noise_ids] = noise
        return labels

    def shift_seq(self, seq, n_classes, shift_size=0.3, noise_strategy="random"):
        update_sample_size = int(len(seq) / n_classes) - 1
        assert update_sample_size * (1 - shift_size) >= 1, \
            "More than 1 sample per class remains unpolluted is required."
        total_update_samples = update_sample_size * n_classes
        num_noise_samples = int(total_update_samples * shift_size)
        num_clean_samples = total_update_samples - num_noise_samples
        clean_ids = np.arange(n_classes) * update_sample_size + np.random.randint(0, update_sample_size, n_classes)
        exclude = np.delete(np.arange(total_update_samples), clean_ids)
        more_ids = np.random.choice(exclude, num_clean_samples - n_classes, replace=False)
        clean_ids = np.concatenate([clean_ids, more_ids])
        noise_ids = np.delete(np.arange(total_update_samples), clean_ids)
        map_noise_ids = noise_ids // update_sample_size + noise_ids
        return map_noise_ids

    def fetch_batch(self, args, mode='train', sample_strategy='random', augment=True):
        n_classes, batch_size, seq_length = args.n_classes, args.batch_size, args.seq_length
        if mode == 'train':
            data = self.train_data
        elif mode == 'test':
            data = self.test_data
        classes = [np.random.choice(list(range(len(data))), replace=False, size=n_classes) for _ in range(batch_size)]
        if sample_strategy == 'random':  # #(sample) per class may not be equal (sec 7)
            seq = np.random.randint(0, n_classes, [batch_size, seq_length])
        elif sample_strategy == 'uniform':  # #(sample) per class are equal
            seq = np.array([np.concatenate([[j] * int(seq_length / n_classes) for j in range(n_classes)])
                            for _ in range(batch_size)])
            if mode == "test" and args.noise_size != 0:
                noise_seq = np.zeros_like(seq)
                for i in range(batch_size):
                    noise_seq[i] = self.noisify_labels(seq[i], n_classes, noise_size=args.noise_size,
                                                       noise_strategy=args.noise_strategy)
            if mode == "test" and args.reduce_spt_size != 0:
                remove_id = []
                for i in range(batch_size):
                    remove_id.append(self.shift_seq(seq[i], n_classes, shift_size=args.reduce_spt_size))

            for i in range(batch_size):
                if mode == "test":
                    seq_i = seq[i, :]
                    seq_length = len(seq_i)
                    qry_ids = np.arange(1, n_classes+1) * int(seq_length / n_classes) - 1
                    qry_perm = np.random.permutation(np.arange(len(qry_ids)))
                    if args.reduce_spt_size != 0:
                        rm_ids = remove_id[i]
                        spt_ids = np.delete(np.arange(seq_length), np.concatenate([qry_ids, rm_ids]))
                        spt_perm = np.random.permutation(np.arange(len(spt_ids)))
                        seq[i, :] = np.concatenate([seq_i[spt_ids][spt_perm], seq_i[qry_ids][qry_perm], seq_i[rm_ids]])
                    else:
                        spt_ids = np.delete(np.arange(seq_length), qry_ids)
                        spt_perm = np.random.permutation(np.arange(len(spt_ids)))
                        seq[i, :] = np.concatenate([seq_i[spt_ids][spt_perm], seq_i[qry_ids][qry_perm]])
                    if args.noise_size != 0:
                        noise_seq_i = noise_seq[i, :]
                        noise_seq[i, :] = np.concatenate([noise_seq_i[spt_ids][spt_perm], noise_seq_i[qry_ids][qry_perm]])

                else:
                    np.random.shuffle(seq[i, :])


            # spt_length = int((seq_length - n_classes) * (1 - args.reduce_spt_size))
            # remain_ids = np.delete(np.arange(seq_length), np.arange(spt_length, seq_length - n_classes))
            # seq = seq[:, remain_ids]
            # args.seq_length = seq_length = spt_length + n_classes
        sample_ids = np.zeros_like(seq)
        for i in range(batch_size):
            for j in range(n_classes):
                class_mask = seq[i] == j
                current_data = data[classes[i][j]]
                id_in_class = np.random.choice(len(current_data), np.count_nonzero(class_mask), replace=False)
                sample_ids[i, class_mask] = id_in_class

        seq_pic = [[self.augment(data[classes[i][j]][sample_ids[i, j]],
                                 only_resize=not augment)
                    for j in seq[i, :]]
                   for i in range(batch_size)]
        if mode == "test" and args.noise_size != 0:
            seq = noise_seq
        if args.label_type == 'one_hot':
            seq_encoded = one_hot_encode(seq, n_classes)
            seq_encoded_shifted = np.concatenate(
                [np.zeros(shape=[batch_size, 1, n_classes]), seq_encoded[:, :-1, :]], axis=1)
        elif args.label_type == 'five_hot':
            label_dict = [[[int(j) for j in list(baseN(i, 5)) + [0] * (5 - len(baseN(i, 5)))]
                           for i in np.random.choice(list(range(5 ** 5)), replace=False, size=n_classes)]
                          for _ in range(batch_size)]
            seq_encoded_ = np.array([[label_dict[b][i] for i in seq[b]] for b in range(batch_size)])
            seq_encoded = np.reshape(one_hot_encode(seq_encoded_, dim=5), newshape=[batch_size, seq_length, -1])
            seq_encoded_shifted = np.concatenate(
                [np.zeros(shape=[batch_size, 1, 25]), seq_encoded[:, :-1, :]], axis=1)
        return seq_pic, seq_encoded_shifted, seq_encoded

    def random_augmentation(self):
        """generates random rotation angel and pixel shift"""
        angle = np.random.uniform(-self.max_angle, self.max_angle, size=1)[0]
        shifts = randint(-self.max_shift, self.max_shift + 1), randint(-self.max_shift, self.max_shift + 1)
        return angle, shifts

    def augment(self, orig_image, only_resize=False):
        if only_resize:
            # only invert and resize the image
            image = np.array(ImageOps.invert(orig_image.convert('L')).resize(self.image_size), dtype=np.float32)
            np_image = np.array(image, dtype=np.float32).reshape((self.image_size[0] * self.image_size[1]))
        else:
            image = np.array(ImageOps.invert(orig_image.convert('L')), dtype=np.float32)
            angle, shifts = self.random_augmentation()
            # Rotate the image
            rotated = np.maximum(np.minimum(rotate(image, angle=angle, mode='nearest', reshape=False), 255), 0)
            # Shift the image
            shifted = shift(rotated, shift=shifts)
            # Resize the image
            resized = np.asarray(resize(shifted, output_shape=self.image_size), dtype=np.float32) / 255.
            np_image = resized.reshape((self.image_size[0] * self.image_size[1]))
        max_value = np.max(np_image)  # normalization is important
        if max_value > 0.:
            np_image = np_image / max_value
        return np_image
