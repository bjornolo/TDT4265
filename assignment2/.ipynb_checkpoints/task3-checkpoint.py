import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images, one_hot_encode, SoftmaxModel
from task2 import SoftmaxTrainer


def main():
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = .02
    batch_size = 32
    neurons_per_layer = [64, 10]
    momentum_gamma = .9  # Task 3 hyperparameter
    shuffle_data = True
    
    use_improved_weight_init = True
    use_improved_sigmoid = True
    use_momentum = True
    use_relu = False
    
    trained=[]
    validation=[]
    labels = [] 
    
    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init,
        use_relu)
    trainer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history, val_history = trainer.train(num_epochs)
    
    trained.append(train_history)
    validation.append(val_history)
    labels.append(f"One hidden layer of 64 nodes")
    print("FIRST")
    
    #ReLU
    neurons_per_layer = [64,10]
    use_improved_sigmoid = False
    use_relu = True
    
    model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init,
        use_relu)
    trainer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history, val_history = trainer.train(num_epochs)
    
    trained.append(train_history)
    validation.append(val_history)
    labels.append(f"One hidden layer of 64 nodes with ReLU")
    print("SECOND")

    """
    #IMPROVED WEIGHTS & SIGMOID
    use_improved_sigmoid = True
    model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init,
        use_relu)
    trainer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history, val_history = trainer.train(num_epochs)
    
    trained.append(train_history)
    validation.append(val_history)
    labels.append(f"IMPROVED WEIGHTS & SIGMOID")
    print("THIRD")
    
    #IMPROVED WEIGHT, SIGMOID & MOMENTUM
    use_momentum = True
    learning_rate = .02
    model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init,
        use_relu)
    trainer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history, val_history = trainer.train(num_epochs)

    trained.append(train_history)
    validation.append(val_history)
    labels.append(f"IMPROVED WEIGHT, SIGMOID & MOMENTUM")
    print("FOURTH")
    """
    plt.figure(figsize=(20, 12))
    plt.subplot(1, 2, 1)
    plt.ylim([0., 0.9])
    for trained_value, label in zip(trained,labels):
        utils.plot_loss(trained_value["loss"],label, npoints_to_average=10)
        #utils.plot_loss(trained_value["accuracy"],label)
    plt.legend()
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Cross Entropy Loss - Average")
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.ylim([0.8, .99])
    for validation_value, label in zip(validation,labels):
        utils.plot_loss(validation_value["accuracy"], label)
        #utils.plot_loss(validation_value["accuracy"],label)
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("task2c_train_loss.png")
    plt.show()    
    
if __name__ == "__main__":
    main()
