import numpy as np
import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images
from trainer import BaseTrainer
from task3a import cross_entropy_loss, SoftmaxModel, one_hot_encode
np.random.seed(0)


def calculate_accuracy(X: np.ndarray, targets: np.ndarray, model: SoftmaxModel) -> float:
    """
    Args:
        X: images of shape [batch size, 785]
        targets: labels/targets of each image of shape: [batch size, 10]
        model: model of class SoftmaxModel
    Returns:
        Accuracy (float)
    """
    # TODO: Implement this function (task 3c)
    logits=model.forward(X)
    #print(logits.shape)
    #print(targets.shape)
    nbr_correct=0
    for picture in range(X.shape[0]):
        for category in range(targets.shape[1]):
            if targets[picture][category]==True and logits[picture][category]>0.5:
                nbr_correct=nbr_correct+1
                #print(nbr_correct)
    accuracy = nbr_correct/X.shape[0]
    return accuracy


class SoftmaxTrainer(BaseTrainer):

    def train_step(self, X_batch: np.ndarray, Y_batch: np.ndarray):
        """
        Perform forward, backward and gradient descent step here.
        The function is called once for every batch (see trainer.py) to perform the train step.
        The function returns the mean loss value which is then automatically logged in our variable self.train_history.

        Args:
            X: one batch of images
            Y: one batch of labels
        Returns:
            loss value (float) on batch
        """
        # TODO: Implement this function (task 3b)
        logits=self.model.forward(X_batch)
        self.model.backward(X_batch,logits, Y_batch)
        self.model.w=self.model.w -(self.learning_rate*self.model.grad)
        loss =cross_entropy_loss(Y_batch, logits)
        
        return loss

    def validation_step(self):
        """
        Perform a validation step to evaluate the model at the current step for the validation set.
        Also calculates the current accuracy of the model on the train set.
        Returns:
            loss (float): cross entropy loss over the whole dataset
            accuracy_ (float): accuracy over the whole dataset
        Returns:
            loss value (float) on batch
            accuracy_train (float): Accuracy on train dataset
            accuracy_val (float): Accuracy on the validation dataset
        """
        # NO NEED TO CHANGE THIS FUNCTION
        logits = self.model.forward(self.X_val)
        loss = cross_entropy_loss(self.Y_val, logits)

        accuracy_train = calculate_accuracy(
            self.X_train, self.Y_train, self.model)
        accuracy_val = calculate_accuracy(
            self.X_val, self.Y_val, self.model)
        return loss, accuracy_train, accuracy_val


def main():
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = 0.01
    batch_size = 128
    l2_reg_lambda = 0
    shuffle_dataset = True

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    # ANY PARTS OF THE CODE BELOW THIS CAN BE CHANGED.
    
    # Intialize model
    model = SoftmaxModel(l2_reg_lambda=1.0)
    # Train model
    trainer = SoftmaxTrainer(
        model, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val,
    )
    train_history, val_history = trainer.train(num_epochs)
    
    print("Final Train Cross Entropy Loss:",
          cross_entropy_loss(Y_train, model.forward(X_train)))
    print("Final Validation Cross Entropy Loss:",
          cross_entropy_loss(Y_val, model.forward(X_val)))
    print("Final Train accuracy:", calculate_accuracy(X_train, Y_train, model))
    print("Final Validation accuracy:", calculate_accuracy(X_val, Y_val, model))
    
    
    
    
    model1 = SoftmaxModel(l2_reg_lambda=0.1)
    trainer = SoftmaxTrainer(
        model1, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_reg01, val_history_reg01 = trainer.train(num_epochs)
    # You can finish the rest of task 4 below this point.
    
    print("Final Train Cross Entropy Loss:",
          cross_entropy_loss(Y_train, model1.forward(X_train)))
    print("Final Validation Cross Entropy Loss:",
          cross_entropy_loss(Y_val, model1.forward(X_val)))
    print("Final Train accuracy:", calculate_accuracy(X_train, Y_train, model1))
    print("Final Validation accuracy:", calculate_accuracy(X_val, Y_val, model1))
    
    model2 = SoftmaxModel(l2_reg_lambda=0.01)
    trainer = SoftmaxTrainer(
        model2, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_reg02, val_history_reg02 = trainer.train(num_epochs)
    # You can finish the rest of task 4 below this point.
    
    print("Final Train Cross Entropy Loss:",
          cross_entropy_loss(Y_train, model2.forward(X_train)))
    print("Final Validation Cross Entropy Loss:",
          cross_entropy_loss(Y_val, model2.forward(X_val)))
    print("Final Train accuracy:", calculate_accuracy(X_train, Y_train, model2))
    print("Final Validation accuracy:", calculate_accuracy(X_val, Y_val, model2))

    model3 = SoftmaxModel(l2_reg_lambda=0.001)
    trainer = SoftmaxTrainer(
        model3, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_reg03, val_history_reg03 = trainer.train(num_epochs)
    # You can finish the rest of task 4 below this point.
    
    print("Final Train Cross Entropy Loss:",
          cross_entropy_loss(Y_train, model3.forward(X_train)))
    print("Final Validation Cross Entropy Loss:",
          cross_entropy_loss(Y_val, model3.forward(X_val)))
    print("Final Train accuracy:", calculate_accuracy(X_train, Y_train, model3))
    print("Final Validation accuracy:", calculate_accuracy(X_val, Y_val, model3))
    
    
    """
    plt.ylim([0.2, .8])
    utils.plot_loss(train_history["loss"],
                    "Training Loss", npoints_to_average=10)
    utils.plot_loss(val_history["loss"], "Validation Loss")
    plt.legend()
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Cross Entropy Loss - Average")
    plt.savefig("task3b_softmax_train_loss.png")
    plt.show()

    # Plot accuracy
    plt.ylim([0.89, .93])
    utils.plot_loss(train_history["accuracy"], "Training Accuracy")
    utils.plot_loss(val_history["accuracy"], "Validation Accuracy")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("task3b_softmax_train_accuracy.png")
    plt.show()
    """
    # Train a model with L2 regularization (task 4b)


    """
    #plt.ylim([0.2, .8])
    utils.plot_loss(train_history_reg01["loss"],
                    "Training Loss", npoints_to_average=10)
    utils.plot_loss(val_history_reg01["loss"], "Validation Loss")
    plt.legend()
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Cross Entropy Loss - Average")
    plt.savefig("task3b_softmax_train_loss.png")
    plt.show()

    # Plot accuracy
    #plt.ylim([0.89, .93])
    utils.plot_loss(train_history_reg01["accuracy"], "Training Accuracy")
    utils.plot_loss(val_history_reg01["accuracy"], "Validation Accuracy")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("task3b_softmax_train_accuracy.png")
    plt.show()
    """
    """
    images0=np.delete(model.w.T, -1, 1)
    images1=np.delete(model1.w.T, -1, 1)
    #fig, cat = plt.subplots(10, 2)
    #fig.suptitle('Lamda=0 Left L=1.0 Right')
    #plt.axis('off')
    for category in range(images0.shape[0]):
        plt.imshow(images0[category].reshape((28,28)), cmap='gray')
        plt.show()
        plt.imshow(images1[category].reshape((28,28)), cmap='gray')
        plt.show()
    
    """
    
    """
    # Plot accuracy
    #plt.ylim([0.8, .95])
    utils.plot_loss(val_history["accuracy"], "Validation Accuracy L=1")
    utils.plot_loss(val_history_reg01["accuracy"], "Validation Accuracy L=0.1")
    utils.plot_loss(val_history_reg02["accuracy"], "Validation Accuracy L=0.01")
    utils.plot_loss(val_history_reg03["accuracy"], "Validation Accuracy L=0.001")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("task3b_softmax_train_accuracy.png")
    plt.show()
    
    """
    l2w=np.array([np.linalg.norm(model.w,2),np.linalg.norm(model1.w,2),np.linalg.norm(model2.w,2),np.linalg.norm(model3.w,2)])
    l2lambda=np.array([1.0,0.1,0.01,0.001])
    plt.plot(l2lambda,l2w, 'o')
    plt.xlabel("Lambda")
    plt.ylabel("L2 w")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
