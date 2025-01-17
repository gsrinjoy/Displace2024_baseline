#!/bin/bash
path=$1
if [ ! -f $path/utt2spk ]; then
 awk '{print $1,$1}' $path/wav.scp > $path/utt2spk
fi
 utils/utt2spk_to_spk2utt.pl \
    $path/utt2spk > $path/spk2utt

source=$path/rttm 

target=$2/data/final.rttm
{
if [ -f $source ]; then
  cp $source $target 
fi
} || {
  echo "Couldnt copy final rttm to $target (Permission issues?). Proceeding anyway..."
}
