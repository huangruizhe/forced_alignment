ql

cd /export/fs04/a12/rhuang/forced_alignment

conda activate /export/fs04/a12/rhuang/anaconda/anaconda3/envs/espnet_gpu
# export PYTHONPATH=/export/fs04/a12/rhuang/icefall_align/:$PYTHONPATH
export PYTHONPATH=/export/fs04/a12/rhuang/icefall/:$PYTHONPATH
export PYTHONPATH=/export/fs04/a12/rhuang/icefall_align/egs/spgispeech/ASR/:$PYTHONPATH

nvidia-smi
source /home/gqin2/scripts/acquire-gpu 1
echo $CUDA_VISIBLE_DEVICES

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

########################################################################
# parameters 

audio=""  # preferable to be a wav file # TODO: look at gengle to see how it is handled
transcript=""
data_dir="/export/fs04/a12/rhuang/contextualizedASR/data/ec53_kaldi_heuristics8_2"

audio_len_max=$((10*60))  # 10 minutes each segment

cuts="$data_dir/cuts.jsonl.gz"
lang_dir="data/lang_letter"
exp_dir="exp/$(basename $data_dir)"

########################################################################

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  log "Stage 1: preparing data"
  
  # audio + text: will convert to kaldi dir first
  if [ -n "$audio" && -n "$transcript" ]; then
    duration=$(ffmpeg -i $audio 2>&1 | grep Duration | sed -nE 's/^.?*Duration:\s(.*?),.*$/\1/p')
    duration=$(echo $duration | awk -F: '{ print ($1 * 3600) + ($2 * 60) + $3 }')
    num_segment=$(echo "($duration+$audio_len_max-1)/$audio_len_max" | bc)

    # wav.scp
    echo "$audio"
    
    # segments

    # text
  fi

  # kaldi dir
  python local2/prepare_lhotse_cutset.py \
    --data-dir "$data_dir"
  
  # lhotse cuts
  ls $data_dir/cuts.jsonl.gz
  cuts=$data_dir/cuts.jsonl.gz
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  log "Stage 2: prepare lang dir"
  mkdir -p ${lang_dir}
  
  python -c "
import lhotse
import re
cuts = lhotse.load_manifest('${cuts}')
p = r'[^\w\’\']'
# words = {'<eps>':0, '!SIL':1, '<SPOKEN_NOISE>':2, '<UNK>':3}
words = {'<eps>':0, '|':1}
for cut in cuts:
  text = cut.supervisions[0].text.split()
  for w in text:
    w = re.sub(p, '', w)
    if w not in words:
      words[w] = len(words)
for i, w in enumerate(words):
  print(f'{w}\t{i}')
" > ${lang_dir}/words.txt

  for disambig_sym in 0 "SKIP" "INS"; do
    next_word_id=$(tail -n1 $lang_dir/words.txt | awk '{print $2 + 1}' | bc)
    echo "#$disambig_sym ${next_word_id}" >> $lang_dir/words.txt
  done

  python local/prepare_lang_letter.py \
    --lang-dir ${lang_dir} \
    --oov '<UNK>'
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  log "Stage 3: do alignment with wav2vec2"
  mkdir -p $exp_dir
  python /export/fs04/a12/rhuang/forced_alignment/align.py \
    --cuts $cuts \
    --exp-dir $exp_dir \
    --lang-dir $lang_dir
fi
  