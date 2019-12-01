import numpy as np

DELTA = np.power(1.0, -15.0)


def check_image(image):
    assert isinstance(image, np.ndarray)
    assert image.shape == (512, 512)
    assert np.amin(image) >= 0 - DELTA and np.amax(image) <= 1 + DELTA


def blur_monitor(blur_function):
    def wrapper(*args, **kwargs):
        assert 1 <= len(args) <= 3
        assert len(kwargs) == 0
        pixel_matrix = args[0]
        if len(args) > 1:
            kernel_size_width = args[1]
            assert kernel_size_width > 0
        if len(args) == 3:
            kernel_size_height = args[2]
            assert kernel_size_height > 0
        check_image(pixel_matrix)
        output = blur_function(*args, **kwargs)
        check_image(output)
        assert np.mean(output) != np.mean(args[0])
        return output

    return wrapper


def noisy_monitor(noisy_function):
    def wrapper(*args, **kwargs):
        assert 1 <= len(args) <= 3
        assert len(kwargs) == 0
        check_image(args[0])
        if len(args) > 1:
            mean = args[1]
            assert np.abs(mean - 0) <= 0.1
        if len(args) == 3:
            stddev = args[2]
            assert stddev >= 0
        output = noisy_function(*args, **kwargs)
        check_image(output)
        assert np.mean(output) != np.mean(args[0])
        return output
    return wrapper


def brightness_monitor(brightness_function):
    def wrapper(*args, **kwargs):
        assert len(args) == 1 or len(args) == 2
        assert len(kwargs) == 0
        check_image(args[0])
        if len(args) == 2:
            brightness_delta = args[1]
            assert 0 <= brightness_delta <= 1
        output = brightness_function(*args, **kwargs)
        check_image(output)
        return output
    return wrapper

