import os 
from Pre_processing import segment_extraction 

#batch extraction of the .pt file

def batch_process (input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    files = os.listdir(input_dir)

    wav_file = []

    for f in files:
        if f.endswith('.wav'):
            wav_file.append(f)
    
    for wav in wav_file:
        base = os.path.splitext(wav)[0]
        txt_file = base + '.txt'

        wav_path = os.path.join(input_dir, wav)
        txt_path = os.path.join(input_dir, txt_file)
        segment_extraction(wav_path, txt_path, output_dir)




if __name__ == '__main__':
    input_folder = r"D:\Resnet\ICBHI_final_database\ICBHI_final_database"
    output_folder = r"D:\Resnet\pt_file_save"
    batch_process(input_folder, output_folder)

