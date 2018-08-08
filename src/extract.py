import os, errno, csv, resampy
import numpy as np
import soundfile as sf
from matplotlib import pyplot as plt
from matplotlib import patches

import vggish_input, params
from boxes import calculate_boxes
from xml_util import write_xml

plt.rcParams['figure.dpi'] = 100
np.random.seed(5239)

file_names = []
audio_file_paths = []
labels = []

# Different slashes for Windows OS
slash = '\\' if os.name == 'nt' else '/'

# Get the audio directory path, place yours here
audio_path = os.path.dirname(params.AUDIO_PATH)

# Get the names and paths of all params.FILE_EXTENSION files in the audio folder
for root, sub_dirs, files in os.walk(audio_path):
    for file in files:
        if file.endswith(params.FILE_EXTENSION):
            file_names.append(file)
            audio_file_paths.append(os.path.join(root, file))

            folder_name = root.split('\\')[-1]

            # Folder names become labels
            if (folder_name != 'data' and folder_name not in labels):
                labels.append(folder_name)

print('Found these labels (from folder names): {},\nnumber of labels: {}'.format(labels, len(labels)))


# Create directories to put spectrograms in if necessary
src_folder = os.path.dirname(os.path.realpath(__file__))
result_root = src_folder + '/../data/'
result_directories = [result_root + 'train/images',
                      result_root + 'test/images',
                      result_root + 'train/annotations',
                      result_root + 'test/annotations']

for path in result_directories:
    try:
        os.makedirs(path)
    except OSError as exception:
        if (exception.errno != errno.EEXIST):
            raise

# Export labels (.txt) for darkflow
text_file = open(result_root + 'labels.txt', 'w')

for label in sorted(labels):
    text_file.write('{}\n'.format(label))

text_file.close()

# Load audio and annotations
# Produce a batch of log mel spectrogram examples for each urbansound_annotation.
total_audio_files = len(audio_file_paths)
print('\nExtracting spectrograms and bounding boxes from {} audio files...\n'.format(total_audio_files))

np.random.shuffle(audio_file_paths)
max_train_index = int(total_audio_files * 0.8)
train, test = audio_file_paths[:max_train_index], audio_file_paths[max_train_index:]


for file_number, audio_file_path in enumerate(audio_file_paths):
    train_or_test_folder = 'train' if audio_file_path in train else 'test'
    audio_data, sample_rate = sf.read(audio_file_path)

    # Convert audio to mono
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)

    # Resample
    if sample_rate != params.SAMPLE_RATE:
        audio_data = resampy.resample(audio_data, sample_rate, params.SAMPLE_RATE)
        sample_rate = params.SAMPLE_RATE

    # Read annotations
    path_no_extension = audio_file_path.split(params.FILE_EXTENSION)[0]
    f = open(path_no_extension + '.csv', 'r')
    csv_annotations_urbansound = csv.reader(f)
    
    # Get an audio snippet for each urbansound_annotation and generate spectrograms 
    for urbansound_annotation in csv_annotations_urbansound:
        start_time_seconds = float(urbansound_annotation[0])
        end_time_seconds = float(urbansound_annotation[1])
        event_label = urbansound_annotation[3]

        start_sample = int(start_time_seconds * sample_rate)
        end_sample = int(end_time_seconds * sample_rate)

        audio_snippet = audio_data[start_sample:end_sample]
        
        input_batch = vggish_input.waveform_to_examples(audio_data, sample_rate)

        for spectrogram_number, spectrogram in enumerate(input_batch):
            # Save Spectrogram image for YOLO training
            spectrogram_for_picture = np.flip(np.rot90(spectrogram, 1), 0)

            boxes = calculate_boxes(spectrogram)

            if boxes:
                file_name = '{}_{}_{}'.format(event_label, path_no_extension.split(slash)[-1], spectrogram_number)
                image_path = os.path.join(result_root, train_or_test_folder, 'images', file_name + '.jpg')

                fig = plt.figure()
                fig.patch.set_visible(False)

                ax = plt.axes([0,0,1,1], frameon=False)

                # Then we disable our xaxis and yaxis completely. If we just say plt.axis('off'),
                # they are still used in the computation of the image padding.
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                # ax = fig.add_subplot(111)
                plt.axis('off')
                plt.imshow(spectrogram_for_picture)

                # extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                image_size = fig.get_size_inches() * fig.dpi

                # plt.savefig(image_path, bbox_inches=extent)
                plt.close()

                def scale_box(box):
                    left, right, bottom, top = box
                    
                    x_max, y_max = spectrogram.shape
                    x_max_new, y_max_new = image_size

                    new_left = int(np.interp(left, [0, x_max], [0, x_max_new]))
                    new_right = int(np.interp(right, [0, x_max], [0, x_max_new]))
                    new_bottom = int(np.interp(bottom, [0, y_max], [0, y_max_new]))
                    new_top = int(np.interp(top, [0, y_max], [0, y_max_new]))

                    return new_left, new_right, new_bottom, new_top

                boxes = list(map(scale_box, boxes))
                
                xml_path = os.path.join(result_root, train_or_test_folder, 'annotations', file_name)
                write_xml(xml_path, image_size, boxes, event_label)

                # # Uncomment this to save presentational plots to '../plots'
                # fig = plt.figure()
                # ax = fig.add_subplot(111)
                # plt.title('Significant Parts of {}'.format(event_label))
                # ax.set_xlabel('Frames (10ms each)')
                # ax.set_ylabel('Frequency Bands')
                # ax.imshow(spectrogram_for_picture, origin='lower')

                # for box in boxes:
                #     left, right, bottom, top = box
                #     width = right - left
                #     height = top - bottom

                #     rectangle = patches.Rectangle((left, bottom), width, height, linewidth=1, edgecolor='r', facecolor='none')
                #     ax.add_patch(rectangle)

                # plt.savefig(os.path.join(src_folder + '/../plots/', file_name))
                # plt.close()

    f.close()
    print('Progress: {} out of {}\n'.format(file_number + 1, total_audio_files))
