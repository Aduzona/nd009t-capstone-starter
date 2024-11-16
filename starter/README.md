**NOTE:** This file is a template that you can use to create the README for your project. The **TODO** comments below will highlight the information you should be sure to include.

# Your Project Title Here

**TODO:** Write a short introduction to your project.

## Project Set Up and Installation
**OPTIONAL:** If your project has any special installation steps, this is where you should put it. To turn this project into a professional portfolio project, you are encouraged to make your `README` detailed and self-explanatory. For instance, here you could explain how to set up your project in AWS and provide helpful screenshots of the process.

## Dataset

### Overview
**TODO**: Explain about the data you are using and where you got it from.

#### Preprocessing in `create_data_loaders`

The `create_data_loaders` function already addresses the variability in image dimensions using the following transformations:
1. **Resize**: All images are resized to 256 pixels on the smaller side while preserving the aspect ratio.
2. **Center Crop**: Images are cropped to a 224x224 square, which is the standard input size for ResNet50.
3. **Normalization**: The pixel values are normalized to match the mean and standard deviation of the ImageNet dataset, which is necessary for pre-trained ResNet50.

These steps ensure that the model receives uniform input images, regardless of their original dimensions, and aligns with ResNet50's input requirements. Therefore, the code is sufficient for handling the dimension variability seen in the dataset.

### Access
**TODO**: Explain how you are accessing the data in AWS and how you uploaded it

## Model Training
**TODO**: What kind of model did you choose for this experiment and why? Give an overview of the types of hyperparameters that you specified and why you chose them. Also remember to evaluate the performance of your model.

## Machine Learning Pipeline
**TODO:** Explain your project pipeline.

## Standout Suggestions
**TODO (Optional):** This is where you can provide information about any standout suggestions that you have attempted.
