import os
import csv

###########################################
# Write the image path and age label to a csv file
#directory_path = "/work3/s212461/data/Small_db"
#output_file_path = "/work3/s212461/data/Small_db_labels.csv"

#if not os.path.exists(directory_path):
#    print(f"Directory {directory_path} does not exist.")
#    exit()

#files = os.listdir(directory_path)

#image_files = [f for f in files if f.lower().endswith((".jpg",".png"))]

#with open(output_file_path, "w", encoding="utf-8") as f:
#    for img in image_files:
#        age_label = img.split("_")[2]
#        f.write(f"{img}, {age_label}\n")

#with open(output_file_path, mode="w", newline='', encoding="utf-8") as file:
#    csv_writer = csv.writer(file)
#    csv_writer.writerow(["image_path", "age"])  # Writing the header (optional)

#    for img in image_files:
        #age_label = img.split("_")[2] #ÍBS: For the AgeDB
#        age_label = img[-6:-4] #ÍBS: For the SmallDB and the Morph database (already preprocessed in another csv for all data)
#        csv_writer.writerow([img, age_label])  # Writing the data

#print(f"Image titles have been saved to {output_file_path}.")

###############################
# Now to the splitting of the database into training and validation

#Step 1: Load and Split the Dataset
#Read the CSV File: Use pandas to read the CSV file.
#Split the Data: Use train_test_split from sklearn.model_selection to split the dataset.

#import pandas as pd
#from sklearn.model_selection import train_test_split

# Read the CSV file
#df = pd.read_csv('/work3/s212461/data/meta_data_final.csv')

# Split the dataset into training and validation sets (80% training, 20% validation)
#train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

#Step 2: Save the Split Data into New CSV Files
# Save the training and validation datasets into separate CSV files
#train_df.to_csv('/work3/s212461/data/meta_data_final_train.csv', index=False)
#val_df.to_csv('/work3/s212461/data/meta_data_final_val.csv', index=False)

#Step 3: Create Folders and Move Images
#Create Directories: Create directories for the training and validation image sets.
#Move Images: Move the images to their respective directories based on the split.

#import os
#import shutil

#os.makedirs('/work3/s212461/data/all_data_small_final_train', exist_ok=True)
#os.makedirs('/work3/s212461/data/all_data_small_final_val', exist_ok=True)

#def move_images(df, source_folder, destination_folder):
#    for _, row in df.iterrows():
#        source_path = os.path.join(source_folder, row['image_path'])
#        destination_path = os.path.join(destination_folder, row['image_path'])
#        shutil.move(source_path, destination_path)

#move_images(train_df, '/work3/s212461/data/all_data_small_final', '/work3/s212461/data/all_data_small_final_train')
#move_images(val_df, '/work3/s212461/data/all_data_small_final', '/work3/s212461/data/all_data_small_final_val')


#################################################
# This is to split the embeddings:
import pandas as pd
import os
import shutil

# Load the split dataframes
train_df = pd.read_csv('/work3/s212461/data/meta_data_final_train.csv')
val_df = pd.read_csv('/work3/s212461/data/meta_data_final_val.csv')

# Create directories for training and validation embeddings
os.makedirs('/work3/s212461/data/face_embeddings_final_train', exist_ok=True)
os.makedirs('/work3/s212461/data/face_embeddings_final_val', exist_ok=True)

def move_embeddings(df, embeddings_source_folder, embeddings_destination_folder):
    for _, row in df.iterrows():
        # Assuming the embedding file is named similarly to the image but with a different extension (e.g., .npy)
        embedding_filename = os.path.splitext(row['image_path'])[0] + '.npy'
        source_path = os.path.join(embeddings_source_folder, embedding_filename)
        destination_path = os.path.join(embeddings_destination_folder, embedding_filename)

        # Move the embedding file
        shutil.move(source_path, destination_path)

# Define the source folder where all embeddings are currently stored
embeddings_source_folder = '/work3/s212461/data/face_embeddings_final'

# Move the embeddings to their respective training and validation directories
move_embeddings(train_df, embeddings_source_folder, '/work3/s212461/data/face_embeddings_final_train')
move_embeddings(val_df, embeddings_source_folder, '/work3/s212461/data/face_embeddings_final_val')


#################################################
# THIS IS JUST TO CHECK THE DATABASES
#import pandas as pd
#from collections import Counter

# Read the CSV file
#df = pd.read_csv('/work3/s212461/data/All_data.csv')

#negative_ages = df[df['age'] < 1]['age']
#if not negative_ages.empty:
#    print("Negative ages found:")
#    print(negative_ages)
#else:
#    print("No negative ages found")

# Group by 'age' and count occurrences
#age_counts = Counter(df['age'])
#print(age_counts)
