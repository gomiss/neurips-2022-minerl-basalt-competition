from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import time
import os
import sys
sys.path.append(os.path.dirname(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(BASE_DIR)
import copy
import presets
import cv2
from waterfall.waterfall import is_in_water

os.environ['TORCH_HOME'] = os.path.dirname(os.path.dirname(__file__))

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception, regnet_x_3_2gf]
model_name = "resnet"
num_classes = 2
batch_size = 8
num_epochs = 500
# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = True


######################################################################
# Helper Functions
# ----------------
#
# Before we write the code for adjusting the models, lets define a few
# helper functions.
#
# Model Training and Validation Code
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The ``train_model`` function handles the training and validation of a
# given model. As input, it takes a PyTorch model, a dictionary of
# dataloaders, a loss function, an optimizer, a specified number of epochs
# to train and validate for, and a boolean flag for when the model is an
# Inception model. The *is_inception* flag is used to accomodate the
# *Inception v3* model, as that architecture uses an auxiliary output and
# the overall model loss respects both the auxiliary output and the final
# output, as described
# `here <https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958>`__.
# The function trains for the specified number of epochs and after each
# epoch runs a full validation step. It also keeps track of the best
# performing model (in terms of validation accuracy), and at the end of
# training returns the best performing model. After each epoch, the
# training and validation accuracies are printed.
#

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = 0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc >= best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                best_epoch = epoch
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f} in epoch {}'.format(best_acc, best_epoch))

    # load best model weights
    model.load_state_dict(best_model_wts)

    # save
    # torch.save(model.state_dict(), f'{model_name}_{num_epochs}_last.pb')
    save_path = os.path.join(BASE_DIR, f'train/{model_name}_{num_epochs}_best.pb')
    torch.save(best_model_wts, save_path)
    print(f"save model to {save_path}")
    return model, val_acc_history


######################################################################
# Set Model Parameters’ .requires_grad attribute
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# This helper function sets the ``.requires_grad`` attribute of the
# parameters in the model to False when we are feature extracting. By
# default, when we load a pretrained model all of the parameters have
# ``.requires_grad=True``, which is fine if we are training from scratch
# or finetuning. However, if we are feature extracting and only want to
# compute gradients for the newly initialized layer then we want all of
# the other parameters to not require gradients. This will make more sense
# later.
#

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0
    crop_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        # model_ft = models.resnet50(pretrained=use_pretrained)
        model_ft = torch.load(os.path.join(BASE_DIR, 'hub/checkpoints/resnet50-0676ba61.pth'))
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[1].in_features
        model_ft.classifier[1] = nn.Linear(num_ftrs, num_classes)
        input_size = 232
        crop_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3 
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299
    elif model_name == "regnet_x_3_2gf":
        model_ft = models.regnet_x_3_2gf(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 256
        crop_size = 224
    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size, crop_size

def generate_ft_model():
    # Initialize the model for this run
    model_ft, input_size, crop_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
    # Send the model to GPU
    model_ft = model_ft.to(device)

    return model_ft, input_size, crop_size


def train_epoch():
    data_dir = os.path.join(BASE_DIR, 'waterfall/waterfall_classify_data')
    # process the given videos to get training data
    from video_process import get_classification_data
    get_classification_data(data_path=os.path.join(BASE_DIR, 'data/MineRLBasaltMakeWaterfall-v0'))

    # os.system(f"""cd {BASE_DIR}/hub && 7z x myfiles.zip.001 -y""")

    model_ft, input_size, crop_size = generate_ft_model()
    # Print the model we just instantiated
    print(model_ft)

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((crop_size, crop_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        'val': transforms.Compose([
            transforms.Resize((crop_size, crop_size)),
            # transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
    }
    # data_transforms = {
    #     'train': presets.ClassificationPresetTrain(crop_size=crop_size),
    #     'val': presets.ClassificationPresetEval(crop_size=crop_size, resize_size=input_size)
    # }

    print("Initializing Datasets and Dataloaders...")

    # Create training and validation datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    # Create training and validation dataloaders
    dataloaders_dict = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in
        ['train', 'val']}


    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t", name)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()

    # Train and evaluate
    model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs,
                                 is_inception=(model_name == "inception"))

class CliffDetector:
    def __init__(self, model_path, save2video=False):
        model, input_size, crop_size = generate_ft_model()
        model.load_state_dict(torch.load(model_path))
        print(f"Load model from {model_path}")
        model.eval()

        self.model = model
        self.pre_process = transforms.Compose([
            transforms.Resize((crop_size, crop_size)),
            # transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        # self.pre_process = presets.ClassificationPresetEval(crop_size=crop_size, resize_size=input_size)

        self.save2video = save2video
        if save2video:
            self.video = cv2.VideoWriter("tmp/waterfall.mp4", 0, 1, (640, 360))

    def check(self, img):
        if self.save2video:
            self.video.write(img)
        # st = time.time()
        with torch.no_grad():
            input_ = transforms.ToPILImage()(img)
            res = self.pre_process(input_).to(device)
            # t1 = time.time()
            res = self.model(res[None, :])
            # print("model time {}, process time {}".format(time.time() - t1, t1 - st))

        prob = torch.softmax(res, -1).tolist()[0]
        if prob[0] > prob[1]:
            print(prob[0])
        if prob[0] < 0.75:
            return False

        # plt.imsave('tmp/terminal_conf{:.2f}_{}.png'.format(prob[0], time.time()), img)
        return True

    def finish_video_record(self):
        cv2.destroyAllWindows()
        self.video.release()

if __name__ == "__main__":
    train_epoch()

    # import cv2
    # cliff_detector = CliffDetector('resnet_best.pb')
    # IMG_PATHS = [
    #     'waterfall_classify_data/val/none/cheeky-cornflower-setter-1308212333fd-20220717-131021-800.png',
    #     'waterfall_classify_data/val/cliff/cheeky-cornflower-setter-1308212333fd-20220717-131021-765.png'
    # ]
    # for IMG_PATH in IMG_PATHS:
    #     img = cv2.imread(IMG_PATH)
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     cliff_detector.check(img)