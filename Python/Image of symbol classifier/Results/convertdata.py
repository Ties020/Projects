import zipfile
import tarfile
import inkml2img2
import xml.etree.ElementTree as ET
import io
import os

print('Start converting...')
images = []
file_names = []
extract_dir = 'temp/' 
with tarfile.open("mathwriting-2024mini.tgz", 'r:gz') as zip_data:
    # zip_data.extractall(path=extract_dir)
    all_content = zip_data.getnames()
    for item in all_content:
        if not item.endswith('/'):
            if item.endswith(('.inkml')):
                file_path = extract_dir + item
                with open(file_path, 'r') as file:
                    try:
                        base_name = os.path.splitext(item)[0]
                        file_names.append(base_name)
                        try:
                            root = ET.fromstring(file.read())
                            image = inkml2img2.inkml_to_image(root)
                            images.append(image)
                        except ET.ParseError as e:
                            print(f"Error parsing XML from file {item}: {e}")
                        except Exception as e:
                            print(f"Error processing file {item}: {e}")
                    except KeyError as e:
                        print(f"File {item} not found or could not be extracted: {e}")
                    except Exception as e:
                        print(f"Unexpected error with file {item}: {e}")

print('Start writing...')
written_files = []
with zipfile.ZipFile('mathwriting-2024mini_imgr.zip', 'w') as zipf:
    for idx, image in enumerate(images):
        try:
            name = file_names[idx]+".png"
            if name not in written_files:
                bytearray = io.BytesIO()
                image.save(bytearray, format='PNG')
                bytearray.seek(0)

                zipf.writestr(name, bytearray.read())
                written_files.append(name)
            else: 
                print(f"Duplicate file skipped: {name}")
        except Exception as e:
            print(f"Error saving image {idx+1}: {e}")

print('Conversion completed!') 
