# Project Proposal

## **Domain Background**

Distribution centers like Amazon Fulfillment Centers process millions of products daily. Accurate inventory tracking is critical to ensure operational efficiency, minimize errors, and enhance customer satisfaction. This project aims to build a machine learning model to classify bin images based on the number of objects they contain, enabling automation in inventory management.

## **Problem Statement**

Inventory tracking at distribution centers requires manual or semi-automated processes, which can be time-consuming and error-prone. This project addresses the problem by automating the classification of bin images into five categories (1–5 objects). By leveraging computer vision, we aim to improve inventory accuracy, reduce manual effort, and enhance overall efficiency.

## **Solution Statement**

This project uses ResNet50 as a pre-trained model, fine-tuned for the multi-class classification task. Images of bins are processed, and metadata is used to label each image with the number of objects (1–5). The model is trained using AWS SageMaker, adhering to best machine learning practices, including data preprocessing, validation, and deployment. The solution is scalable and can be deployed as a real-time inference system to classify images directly from distribution centers.

## **Datasets and Inputs**

The dataset comprises images of bins from the **Amazon Bin Image Dataset**, with metadata indicating the `EXPECTED_QUANTITY` (number of objects in the bin). The images are labeled into five classes (1–5 objects). Key characteristics:
1. Images vary in dimensions and are resized to 224x224 for uniformity.
2. Dataset split: 70% training, 20% validation, 10% testing.
3. The dataset is stored and accessed via AWS S3.

## **Benchmark Model**


### **Benchmark Model Selection**
ResNet50 pre-trained on ImageNet serves as the benchmark model. It provides a baseline for performance comparison and is expected to achieve high accuracy on this task due to transfer learning. It is chosen because:
1. **Transfer Learning**: ResNet50 is a proven architecture for image classification tasks and benefits from pre-trained weights on ImageNet.
2. **Efficiency**: It strikes a balance between model complexity and performance, making it suitable for medium-scale datasets like this.
3. **Scalability**: ResNet50 can generalize well for the task of object counting in bins after fine-tuning.

### **Why ResNet50 is a Good Benchmark**
- Provides a baseline to evaluate the custom-trained model.
- Can be fine-tuned to learn features specific to this dataset.

## **Evaluation Metrics**

Since this is a multi-class classification problem (5 classes: 1–5 objects), The model’s performance is evaluated using:

1. **Accuracy**: To measure overall performance
   - Measures the percentage of correctly classified images.
   - Formula:  
     \[
     \text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Number of Predictions}}
     \]

2. **Confusion Matrix**: To identify specific misclassifications
   - Helps visualize the model’s performance by showing the number of true positives, false positives, and false negatives for each class.

3. **Precision, Recall, and F1-Score (Per Class)**: To evaluate per-class performance
   - Precision: Measures the proportion of true positives out of all predicted positives for each class.
   - Recall: Measures the proportion of true positives out of all actual positives for each class.
   - F1-Score: Harmonic mean of precision and recall, balancing the tradeoff between them.

These metrics provide a holistic view of the model’s performance beyond just accuracy, especially when classes may have imbalanced data.

## **Project Design**

1. **Data Preprocessing**:
   - Images are resized to 224x224 and normalized.
   - Dataset split into training, validation, and test sets.

2. **Model Training**:
   - Fine-tune ResNet50 on the processed dataset.
   - Use `CrossEntropyLoss` for multi-class classification.
   - Monitor training and validation loss/accuracy.

3. **Model Deployment**:
   - Deploy the trained model to a SageMaker endpoint.
   - Perform real-time inference on sample images.

4. **Optional Extensions**:
   - Hyperparameter tuning for optimal performance.
   - Model debugging and profiling using SageMaker tools.

## **Expected Outcome**

- A trained ResNet50 model capable of accurately classifying bin images into five categories.
- Deployment-ready solution for real-time inference in distribution centers.

## **Future Scope**

- Extend the system to handle bins with more than five objects.
- Integrate with robotic systems for automated inventory management.
- Explore lightweight models for edge device deployment.