# face-projection

This projects main goal is to have a simple library to project information into faces while retaining the facial structure.


## Description

Using the Canonical Face Model provided by Google, we can draw any UV map inside the given boundaries to the face.
This allows us to visualize certain properties of facial features.
The user can provide `face_data` which then projected onto the `face_img`.
We support automatic face detection with landmark extraction using `mediapipe`.
However, the user can also provide the landmarks manually if they are already known and fit the Canonical Face Model.

## Installation

### MacOS

To use meshpy a C++ compiler is required.
On MacOS you can use xcodebuild which is installed by default.
To enable it please run the following command and agree to the license agreement.
Then it should be able to compile the C++ code in your python environment.

```bash
sudo xcodebuild -license
```

## Usage

```python
import face_projection as fp

warper = fp.Warper()

face_img = load_image("face.jpg") # load image of face with your favorite library
face_data = warper.create_canvas() # creates a empty image with the current size of the face model

# draw something on the face_data
# your implementation here

# project the face_data onto the face_img, using the automatic face detection
# beta is the blending factor between 0 and 1
warped_face = warper.apply(face_img, face_data, beta=0.2)

# if you already know the landmarks, you can provide them manually
warped_face = warper.apply(face_img, face_data, landmarks=landmarks, beta=0.2)
```

## Future Work
- [ ] The user can provide a mask to disable certain parts of the projection
- [ ] Sample generators for different face properties
- [ ] More face models
- [ ] Custom face models
- [ ] Canvas fitter to fit to the general face model


## References

## Citation

If you use our work, please cite our paper:

```bibtex
TODO :^)
```
