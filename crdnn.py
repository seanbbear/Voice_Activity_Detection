from pydub.audio_segment import AudioSegment
from speechbrain.pretrained import VAD
import torchaudio
import os

DATA_PATH = "/home/twswaxx438/audio_crawler/data/wav/denoise/"


def main():
    for file in os.listdir(DATA_PATH):
        if file.endswith("wav"):
            try:
                segment_wav(DATA_PATH, file)
            except Exception as e:
                print(str(e))
    # segment_wav("/home/twswaxx438/audio_crawler/data/wav/denoise/", "202001011986083.wav")


def segment_wav(path, file):
    wav_file = path + file
    vad = VAD.from_hparams(source="speechbrain/vad-crdnn-libriparty", savedir="pretrained_models/vad-crdnn-libriparty")
    boundaries = vad.get_speech_segments(wav_file, overlap_small_chunk=True, apply_energy_VAD=True, len_th=2)
    
    # 去掉太短的段落(<2.5秒)
    boundaries = vad.remove_short_segments(boundaries, len_th=2.5)
    
    sound = AudioSegment.from_wav(wav_file)
    
    
    print(f"Length of Segmented data : {len(boundaries)}")
    
    # 若切不出來 直接return檔案
    # if len(boundaries) == 0:
    #     sound.export("./segmented_wav/" + file)
    
    for i in range(len(boundaries)):
        # 時間以毫秒為單位
        begin_time = int(boundaries[i][0]*1000)
        end_time = int(boundaries[i][1]*1000)

        duration = end_time-begin_time
        if duration<(10*1000):
            seg_sound = sound[begin_time:end_time]
            target_path = DATA_PATH + "segmented_wav/" + file[:-4] + "-" +  str(i) + ".wav"
            print(f"export segment file {i} from {file[:-4]} 共{duration/1000}秒")
            seg_sound.export(target_path, format="wav")

if __name__ == "__main__":
    main()
