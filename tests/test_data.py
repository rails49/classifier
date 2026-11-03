import numpy as np

from classifier.data import folds


def test_a_placement_across_archives_goes_with_the_archive_of_its_first_image():
    # Placement 23 spans 10-29 IMG_2902 and 10-29b IMG_2903, listed here 10-29b first.
    archive = np.array(["10-29b", "10-29", "10-29", "10-29b", "10-28"])
    image = np.array(["IMG_2903", "IMG_2902", "IMG_2901", "IMG_2910", "IMG_2849"])
    placement = np.array([23, 23, 22, 24, 0])

    assert folds(archive, image, placement).tolist() == [
        "10-29",
        "10-29",
        "10-29",
        "10-29b",
        "10-28",
    ]
