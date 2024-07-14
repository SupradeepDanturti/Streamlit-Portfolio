from SpeakerCounter import SpeakerCounter
wav_path = "./sample_audio1.wav"
save_dir = "./sample_inference_run2/"
model_path = "ecapa_tdnn"  # ./xvector

audio_classifier = SpeakerCounter.from_hparams(source=model_path, savedir=save_dir)

audio_classifier.classify_file(wav_path)