#

OUTPATH=$PWD/data/weights

mkdir -p $OUTPATH

URLSWECLIP=https://www.dropbox.com/s/s77xw5308jeljlp/Swedish-500k%20Linear%20Weights.pkl
wget -c "${URLSWECLIP}" -P $OUTPATH

URLSWECLIP2M=https://www.dropbox.com/s/82c54rsvlry3kwh/Swedish-2M%20Linear%20Weights.pkl
wget -c "${URLSWECLIP2M}" -P $OUTPATH

URLMCLIP=https://www.dropbox.com/s/oihqzctnty5e9kk/M-BERT%20Distil%2040%20Linear%20Weights.pkl
wget -c "${URLMCLIP}" -P $OUTPATH

URLMCLIPBASE=https://www.dropbox.com/s/y4pycinv0eapeb3/M-BERT-Base-69%20Linear%20Weights.pkl
wget -c "${URLMCLIPBASE}" -P $OUTPATH

