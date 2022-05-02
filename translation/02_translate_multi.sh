for filename in $(ls data_multi);
do
    python 02_translate_multilingual.py --filename $filename
done