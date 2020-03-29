import numbers
import random

from torchvision.transforms import Compose, Lambda
from torchvision.transforms import functional as F


class GroupColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image
    See torchvision.transforms.ColorJitter
    Add a random grayscale
    """

    def __init__(
            self,
            brightness=0.4, contrast=0.4,
            saturation=0.4, hue=0.4,
            grayscale=0.3):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.grayscale = grayscale

    def _get_transform(self):
        curr_brightness = random.uniform(1 - self.brightness, 1 + self.brightness)
        curr_contrast = random.uniform(1 - self.contrast, 1 + self.contrast)
        curr_saturation = random.uniform(1 - self.saturation, 1 + self.saturation)
        curr_hue = random.uniform(-self.hue, self.hue)

        transforms = []
        transforms.append(Lambda(lambda img: F.adjust_brightness(img, curr_brightness)))
        transforms.append(Lambda(lambda img: F.adjust_contrast(img, curr_contrast)))
        transforms.append(Lambda(lambda img: F.adjust_saturation(img, curr_saturation)))
        transforms.append(Lambda(lambda img: F.adjust_hue(img, curr_hue)))
        random.shuffle(transforms)
        curr_grayscale = random.uniform(0, 1)
        if curr_grayscale < self.grayscale:
            transforms.append(
                Lambda(lambda img: F.to_grayscale(
                    img, num_output_channels=3)))
        transform = Compose(transforms)
        return transform

    def __call__(self, img_group):
        transform = self._get_transform()
        out = [transform(img) for img in img_group]
        return out


class ColorJitterVideo(object):
    """Randomly change the brightness, contrast and saturation of an video.

    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, grayscale=0):
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)
        self.grayscale = grayscale

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_params(brightness, contrast, saturation, hue, grayscale):
        """Get a randomized transform to be applied on image.

        Arguments are same as that of __init__.

        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []

        if brightness is not None:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            transforms.append(Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))

        if contrast is not None:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            transforms.append(Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))

        if saturation is not None:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            transforms.append(Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))

        if hue is not None:
            hue_factor = random.uniform(hue[0], hue[1])
            transforms.append(Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        random.shuffle(transforms)

        if grayscale is not None:
            grayscale_factor = random.uniform(0, 1)
            if grayscale_factor < grayscale:
                transforms.append(Lambda(lambda img: F.to_grayscale(img, num_output_channels=3)))

        transform = Compose(transforms)

        return transform

    def __call__(self, frames):
        """
        Args:
            frames (List[PIL Image]): Input frames.

        Returns:
            List[PIL Image]: Color jittered frames.
        """
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue, self.grayscale)
        for frame in frames:
            yield transform(frame)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        format_string += ', grayscale={0})'.format(self.grayscale)
        return format_string
