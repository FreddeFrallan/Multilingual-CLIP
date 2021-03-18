#

OUTPATH=$PWD/data/weights

mkdir -p $OUTPATH

URLSWECLIP=https://www.dropbox.com/s/s77xw5308jeljlp/Swedish-500k%20Linear%20Weights.pkl
wget -c "${URLSWECLIP}" -P $OUTPATH

URLMCLIP=https://www.dropbox.com/s/r5y6ra150b8hb1o/M-BERT%20Distil%2040%20Linear%20Weights.pkl
wget -c "${URLMCLIP}" -P $OUTPATH

