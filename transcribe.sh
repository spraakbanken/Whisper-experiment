#!/bin/bash
for model_name in stable_ts openai kb-whisper; do
	for model_size in tiny small medium large; do
		find ~/tmp/trip_to_stockholm/ -iname *.wav -exec python src/transcribe.py $model_name $model_size cuda {} \+
	done
done