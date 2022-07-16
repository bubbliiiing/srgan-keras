import math
import multiprocessing
import random
import threading
import time
from abc import abstractmethod
from contextlib import closing
from multiprocessing.pool import ThreadPool
from random import randint, shuffle

import cv2
import keras
import numpy as np
import six
from PIL import Image

try:
    import queue
except ImportError:
    import Queue as queue

from .utils import cvtColor, preprocess_input


def get_new_img_size(width, height, img_min_side=600):
    if width <= height:
        f = float(img_min_side) / width
        resized_height = int(f * height)
        resized_width = int(img_min_side)
    else:
        f = float(img_min_side) / height
        resized_width = int(f * width)
        resized_height = int(img_min_side)

    return resized_width, resized_height

class SRganDataset(keras.utils.Sequence):
    def __init__(self, annotation_lines, lr_shape, hr_shape, batch_size):
        super(SRganDataset, self).__init__()

        self.annotation_lines   = annotation_lines
        self.length             = len(annotation_lines)

        self.lr_shape       = lr_shape
        self.hr_shape       = hr_shape

        self.batch_size     = batch_size

    def __len__(self):
        return math.ceil(self.length / float(self.batch_size))

    def on_epoch_end(self):
        shuffle(self.annotation_lines)

    def __getitem__(self, index):
        images_l = []
        images_h = []
        for i in range(index * self.batch_size, (index + 1) * self.batch_size):  
            i = i % self.length

            image_origin = Image.open(self.annotation_lines[i].split()[0])
            if self.rand()<.5:
                img_h = self.get_random_data(image_origin, self.hr_shape)
            else:
                img_h = self.random_crop(image_origin, self.hr_shape[1], self.hr_shape[0])
            img_l = img_h.resize((self.lr_shape[1], self.lr_shape[0]), Image.BICUBIC)

            img_h = preprocess_input(np.array(img_h, dtype=np.float32))
            img_l = preprocess_input(np.array(img_l, dtype=np.float32))

            images_h.append(img_h)
            images_l.append(img_l)
        return np.array(images_l), np.array(images_h)

    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a

    def get_random_data(self, image, input_shape, jitter=.3, hue=.1, sat=1.5, val=1.5, random=True):
        #------------------------------#
        #   读取图像并转换成RGB图像
        #------------------------------#
        image   = cvtColor(image)
        #------------------------------#
        #   获得图像的高宽与目标高宽
        #------------------------------#
        iw, ih  = image.size
        h, w    = input_shape

        if not random:
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            dx = (w-nw)//2
            dy = (h-nh)//2

            #---------------------------------#
            #   将图像多余的部分加上灰条
            #---------------------------------#
            image       = image.resize((nw,nh), Image.BICUBIC)
            new_image   = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data  = np.array(new_image, np.float32)

            return image_data

        #------------------------------------------#
        #   对图像进行缩放并且进行长和宽的扭曲
        #------------------------------------------#
        new_ar = w/h * self.rand(1-jitter,1+jitter)/self.rand(1-jitter,1+jitter)
        scale = self.rand(1, 1.5)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)

        #------------------------------------------#
        #   将图像多余的部分加上灰条
        #------------------------------------------#
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_image.paste(image, (dx, dy))
        image = new_image

        #------------------------------------------#
        #   翻转图像
        #------------------------------------------#
        flip = self.rand()<.5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
        rotate = self.rand()<.5
        if rotate: 
            angle = np.random.randint(-15,15)
            a,b = w/2,h/2
            M = cv2.getRotationMatrix2D((a,b),angle,1)
            image = cv2.warpAffine(np.array(image), M, (w,h), borderValue=[128,128,128]) 

        #------------------------------------------#
        #   色域扭曲
        #------------------------------------------#
        hue = self.rand(-hue, hue)
        sat = self.rand(1, sat) if self.rand()<.5 else 1/self.rand(1, sat)
        val = self.rand(1, val) if self.rand()<.5 else 1/self.rand(1, val)
        x = cv2.cvtColor(np.array(image,np.float32)/255, cv2.COLOR_RGB2HSV)
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:,:, 0]>360, 0] = 360
        x[:, :, 1:][x[:, :, 1:]>1] = 1
        x[x<0] = 0
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)*255
        return Image.fromarray(np.uint8(image_data))

    def random_crop(self, image, width, height):
        #--------------------------------------------#
        #   如果图像过小无法截取，先对图像进行放大
        #--------------------------------------------#
        if image.size[0] < self.hr_shape[1] or image.size[1] < self.hr_shape[0]:
            resized_width, resized_height = get_new_img_size(width, height, img_min_side=np.max(self.hr_shape))
            image = image.resize((resized_width, resized_height), Image.BICUBIC)

        #--------------------------------------------#
        #   随机截取一部分
        #--------------------------------------------#
        width1  = randint(0, image.size[0] - width)
        height1 = randint(0, image.size[1] - height)

        width2  = width1 + width
        height2 = height1 + height

        image   = image.crop((width1, height1, width2, height2))
        return image


#----------------------------------------------------------#
#   多进程进行数据读取的代码，Copy From Keras==2.1.5
#   Training-related part of the Keras engine.
#----------------------------------------------------------#
_SHARED_SEQUENCES = {}
_SEQUENCE_COUNTER = None


def init_pool(seqs):
    global _SHARED_SEQUENCES
    _SHARED_SEQUENCES = seqs


def get_index(uid, i):
    """Get the value from the Sequence `uid` at index `i`.

    To allow multiple Sequences to be used at the same time, we use `uid` to
    get a specific one. A single Sequence would cause the validation to
    overwrite the training Sequence.

    # Arguments
        uid: int, Sequence identifier
        i: index

    # Returns
        The value at index `i`.
    """
    return _SHARED_SEQUENCES[uid][i]


class SequenceEnqueuer(object):
    """Base class to enqueue inputs.

    The task of an Enqueuer is to use parallelism to speed up preprocessing.
    This is done with processes or threads.

    # Examples

    ```python
        enqueuer = SequenceEnqueuer(...)
        enqueuer.start()
        datas = enqueuer.get()
        for data in datas:
            # Use the inputs; training, evaluating, predicting.
            # ... stop sometime.
        enqueuer.close()
    ```

    The `enqueuer.get()` should be an infinite stream of datas.

    """

    @abstractmethod
    def is_running(self):
        raise NotImplementedError

    @abstractmethod
    def start(self, workers=1, max_queue_size=10):
        """Starts the handler's workers.

        # Arguments
            workers: number of worker threads
            max_queue_size: queue size
                (when full, threads could block on `put()`).
        """
        raise NotImplementedError

    @abstractmethod
    def stop(self, timeout=None):
        """Stop running threads and wait for them to exit, if necessary.

        Should be called by the same thread which called start().

        # Arguments
            timeout: maximum time to wait on thread.join()
        """
        raise NotImplementedError

    @abstractmethod
    def get(self):
        """Creates a generator to extract data from the queue.

        Skip the data if it is `None`.

        # Returns
            Generator yielding tuples `(inputs, targets)`
                or `(inputs, targets, sample_weights)`.
        """
        raise NotImplementedError


class OrderedEnqueuer(SequenceEnqueuer):
    """Builds a Enqueuer from a Sequence.

    Used in `fit_generator`, `evaluate_generator`, `predict_generator`.

    # Arguments
        sequence: A `keras.utils.data_utils.Sequence` object.
        use_multiprocessing: use multiprocessing if True, otherwise threading
        shuffle: whether to shuffle the data at the beginning of each epoch
    """

    def __init__(self, sequence,
                 use_multiprocessing=False,
                 shuffle=False):
        self.sequence = sequence
        self.use_multiprocessing = use_multiprocessing

        global _SEQUENCE_COUNTER
        if _SEQUENCE_COUNTER is None:
            try:
                _SEQUENCE_COUNTER = multiprocessing.Value('i', 0)
            except OSError:
                # In this case the OS does not allow us to use
                # multiprocessing. We resort to an int
                # for enqueuer indexing.
                _SEQUENCE_COUNTER = 0

        if isinstance(_SEQUENCE_COUNTER, int):
            self.uid = _SEQUENCE_COUNTER
            _SEQUENCE_COUNTER += 1
        else:
            # Doing Multiprocessing.Value += x is not process-safe.
            with _SEQUENCE_COUNTER.get_lock():
                self.uid = _SEQUENCE_COUNTER.value
                _SEQUENCE_COUNTER.value += 1

        self.shuffle = shuffle
        self.workers = 0
        self.executor_fn = None
        self.queue = None
        self.run_thread = None
        self.stop_signal = None

    def is_running(self):
        return self.stop_signal is not None and not self.stop_signal.is_set()

    def start(self, workers=1, max_queue_size=10):
        """Start the handler's workers.

        # Arguments
            workers: number of worker threads
            max_queue_size: queue size
                (when full, workers could block on `put()`)
        """
        if self.use_multiprocessing:
            self.executor_fn = lambda seqs: multiprocessing.Pool(workers,
                                                                 initializer=init_pool,
                                                                 initargs=(seqs,))
        else:
            # We do not need the init since it's threads.
            self.executor_fn = lambda _: ThreadPool(workers)
        self.workers = workers
        self.queue = queue.Queue(max_queue_size)
        self.stop_signal = threading.Event()
        self.run_thread = threading.Thread(target=self._run)
        self.run_thread.daemon = True
        self.run_thread.start()

    def _wait_queue(self):
        """Wait for the queue to be empty."""
        while True:
            time.sleep(0.1)
            if self.queue.unfinished_tasks == 0 or self.stop_signal.is_set():
                return

    def _run(self):
        """Submits request to the executor and queue the `Future` objects."""
        sequence = list(range(len(self.sequence)))
        self._send_sequence()  # Share the initial sequence
        while True:
            if self.shuffle:
                random.shuffle(sequence)

            with closing(self.executor_fn(_SHARED_SEQUENCES)) as executor:
                for i in sequence:
                    if self.stop_signal.is_set():
                        return
                    self.queue.put(
                        executor.apply_async(get_index, (self.uid, i)), block=True)

                # Done with the current epoch, waiting for the final batches
                self._wait_queue()

                if self.stop_signal.is_set():
                    # We're done
                    return

            # Call the internal on epoch end.
            self.sequence.on_epoch_end()
            self._send_sequence()  # Update the pool

    def get(self):
        """Creates a generator to extract data from the queue.

        Skip the data if it is `None`.

        # Yields
            The next element in the queue, i.e. a tuple
            `(inputs, targets)` or
            `(inputs, targets, sample_weights)`.
        """
        try:
            while self.is_running():
                inputs = self.queue.get(block=True).get()
                self.queue.task_done()
                if inputs is not None:
                    yield inputs
        except Exception as e:
            self.stop()
            six.raise_from(StopIteration(e), e)

    def _send_sequence(self):
        """Send current Sequence to all workers."""
        # For new processes that may spawn
        _SHARED_SEQUENCES[self.uid] = self.sequence

    def stop(self, timeout=None):
        """Stops running threads and wait for them to exit, if necessary.

        Should be called by the same thread which called `start()`.

        # Arguments
            timeout: maximum time to wait on `thread.join()`
        """
        self.stop_signal.set()
        with self.queue.mutex:
            self.queue.queue.clear()
            self.queue.unfinished_tasks = 0
            self.queue.not_full.notify()
        self.run_thread.join(timeout)
        _SHARED_SEQUENCES[self.uid] = None
