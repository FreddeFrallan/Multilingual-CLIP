#

OUTPATH=$PWD/data/weights

mkdir -p $OUTPATH

URLSWECLIP=https://www.dropbox.com/s/g3y6zlp0dwwqk9d/Swe-CLIP%20Linear%20Weights.pkl
wget -c "${URLSWECLIP}" -P $OUTPATH

URLMCLIP=https://www.dropbox.com/s/1bo8ehc4hh7bghm/M-CLIP%20Distil%2040%20Linear%20Weights.pkl
wget -c "${URLMCLIP}" -P $OUTPATH

