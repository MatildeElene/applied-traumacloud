from wordcloud import WordCloud
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Read the transposed CSV file
df_transposed = pd.read_csv('transposed_war_trauma5.csv', index_col=0)

# Sort the dataframe by the 'miscarriage' column in descending order
df_sorted = df_transposed.sort_values(by='war_trauma', ascending=False)

# Convert the dataframe to a dictionary with words and their frequencies
data = df_sorted['war_trauma'].to_dict()

# Function to limit the text to a maximum of 200 characters
def limit_text_length(data, max_chars=200):
    limited_data = {}
    total_chars = 0
    for word, freq in data.items():
        if total_chars + len(word) > max_chars:
            break
        limited_data[word] = freq
        total_chars += len(word) + 1  # +1 for the space or separation between words
    return limited_data

# Limit the text length to 200 characters
limited_data = limit_text_length(data, max_chars=200)

# Print the truncated list of words and their frequencies
print("Truncated list of words and their frequencies (up to 200 characters):")
for word, freq in limited_data.items():
    print(f"{word}: {freq:.4f}")

# Load the cloud mask
cloud_mask = np.array(Image.open('/Users/matildeelene/Desktop/Applied/traumacloud/Wordcloud_mask/cloud_mask.jpeg').convert('L'))

# Function to use off-white color for words
def off_white_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    return "rgb(245, 245, 245)"

# Function to generate the word cloud
def generate_better_wordcloud(data, title, mask=None, color_func=None):
    # Generate the word cloud
    cloud = WordCloud(scale=3,
                      max_words=30,
                      mask=mask,
                      color_func=color_func,
                      background_color=None,  # Transparent background
                      mode='RGBA',  # Use RGBA for transparency
                      collocations=True).generate_from_frequencies(data)
    
    # Convert word cloud to image with transparent background
    image = cloud.to_image()
    image = image.convert("RGBA")
    
    # Make the background transparent
    datas = image.getdata()
    new_data = []
    for item in datas:
        # Change all white (255, 255, 255) pixels to transparent
        if item[:3] == (255, 255, 255):
            new_data.append((255, 255, 255, 0))
        else:
            new_data.append(item)
    
    image.putdata(new_data)
    
    # Display the word cloud
    plt.figure(figsize=(20, 8))
    plt.imshow(image, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.show()
    
    # Save the word cloud to a file with transparent background
    image.save('/Users/matildeelene/Desktop/Applied/traumacloud/wordclouds/text_only_war_trauma5.png', "PNG")

# Call the function to generate and display the word cloud with limited text length
generate_better_wordcloud(limited_data, "war_trauma5", mask=cloud_mask, color_func=off_white_color_func)
