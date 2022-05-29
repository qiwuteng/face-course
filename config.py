root = './data/FS2K/sketch'
train_json_path = './data/FS2K/anno_train.json'
test_json_path = './data/FS2K/anno_test.json'
attributes = ['hair', 'gender', 'earring', 'smile', 'frontal_face']
num_workers = 8
epochs = 100
batch_szie = 16
lr = 1e-3
model = 'VGG19'
model_input_dim = {'AlexNet': 256 * 6 * 6, 'VGG16': 512 * 7 * 7, 'VGG19': 512 * 7 * 7, 'ResNet18': 512, 'ResNet50': 2048} 
checkpoint_path = './checkpoint/'
cm_path = './confusion_matrix'