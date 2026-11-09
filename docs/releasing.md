# Releasing the package

A tag `vX.Y.Z` on `main` publishes `rails49-classifier` to PyPI with version
`X.Y.Z`, then makes a GitHub Release with generated notes that name the model.

1. In a pull request, set the version in `pyproject.toml`.
2. Merge it.
3. Tag the merge commit on `main` and push the tag:

   ```sh
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```

The [release workflow](../.github/workflows/release.yml) publishes nothing for a
tag that is not `vX.Y.Z`, not on `main`, or not the package's version. It
publishes the wheel that `bin/check-wheel.sh` scored a frame with.

If the publish fails, re-run that job. PyPI does not accept a version twice, so a
version that was published stays published.

## The first release

PyPI needs a trusted publisher before the first tag. Add a pending publisher at
<https://pypi.org/manage/account/publishing/>:

| Field | Value |
| :--- | :--- |
| PyPI project name | `rails49-classifier` |
| Owner | `rails49` |
| Repository name | `classifier` |
| Workflow name | `release.yml` |
| Environment name | `release` |

## The model

The wheel holds one model, in `src/classifier/model/`: an ONNX file and the JSON
file `uv run train` writes beside it. To change it, replace both files and set
`MODEL` in `src/classifier/runtime.py`.

A release that changes the model states, in its notes, false clear and false
occupied at the packaged threshold, with 95% intervals from resampling
placements. They are counted per (held-out image, position) along all labeled
track, and given overall, by region and by distance from the image center. The
goals are false clear ≤ 0.1% and false occupied ≤ 1%. A model that misses them
can still be released. The notes say that the numbers hold for one layout and
one set of stock, and that the corpus has too few independent train ends to
confirm 0.1% false clear.
