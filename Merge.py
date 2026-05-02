class TextPreprocessor:
    def __init__(self):
        factory = StemmerFactory()
        self.stemmer = factory.create_stemmer()
        stop_factory = StopWordRemoverFactory()
        self.stopword_remover = stop_factory.create_stop_word_remover()
        #self.translator = Translator()
        self.normalization_dict = {
            'gk': 'tidak', 'ga': 'tidak', 'tdk': 'tidak',
            'dgn': 'dengan', 'utk': 'untuk', 'krn': 'karena',
            'yg': 'yang', 'sy': 'saya', 'klo': 'kalau',
            'tp': 'tapi', 'bs': 'bisa', 'blm': 'belum',
            'sdh': 'sudah', 'lg': 'lagi', 'aj': 'saja'
        }

    def normalize_text(self, text):
            """Normalisasi kata"""
            if pd.isna(text):
                return ""
            text = str(text).lower()
            words = text.split()
            normalized_words = [self.normalization_dict.get(word, word) for word in words]
            return ' '.join(normalized_words)

    def translate_mixed_text(self, text):
        """Translate bahasa campuran ke bahasa Indonesia (TETAP SERIAL)"""
        try:
            if pd.isna(text) or text == "":
                return ""
            translator = Translator()
            detected = translator.detect(text)
            if detected.lang != 'id' and detected.confidence > 0.7:
                translated = translator.translate(text, dest='id')
                return translated.text
            return text
        except Exception as e:
            print(f"Translation error for text: '{text[:50]}...' - {e}")
            return text

    def demojize_text(self, text):
        """Convert emoji ke text"""
        if pd.isna(text):
            return ""
        return emoji.demojize(text, language='id')

    def clean_text(self, text):
        """Pembersihan teks umum"""
        if pd.isna(text):
            return ""
        text = str(text)
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def stem_text(self, text):
        """Stemming menggunakan Sastrawi"""
        if pd.isna(text) or text == "":
            return ""
        return self.stemmer.stem(text)

    def preprocess_text(self, text):
        """Pipeline preprocessing lengkap (SERIAL)"""
        text = self.normalize_text(text)
        text = self.translate_mixed_text(text)
        text = self.demojize_text(text)
        text = self.clean_text(text)
        text = self.stem_text(text)
        return text

    def preprocess_text_parallel(self, series):
        """Pipeline preprocessing lengkap (PARALEL)"""
        normalization_dict = self.normalization_dict
        stemmer_instance = self.stemmer

        def normalize_text_local(text):
             if pd.isna(text):
                 return ""
             text = str(text).lower()
             words = text.split()
             normalized_words = [normalization_dict.get(word, word) for word in words]
             return ' '.join(normalized_words)

        def clean_text_local(text):
             if pd.isna(text):
                 return ""
             text = str(text)
             text = re.sub(r'[^a-zA-Z\s]', ' ', text)
             text = re.sub(r'\s+', ' ', text)
             return text.strip()

        def stem_text_local(text):
             if pd.isna(text) or text == "":
                 return ""
             return stemmer_instance.stem(text)

        def demojize_text_local(text):
             """Convert emoji ke text"""
             if pd.isna(text):
                 return ""
             return emoji.demojize(text, language='id')

        print("Starting parallel preprocessing (Normalize, Demojize, Clean, Stem)...")
        processed_series = series.parallel_apply(normalize_text_local)
        processed_series = processed_series.parallel_apply(demojize_text_local)
        processed_series = processed_series.parallel_apply(clean_text_local)
        processed_series = processed_series.parallel_apply(stem_text_local)
        print("Parallel preprocessing finished.")

        return processed_series

    def preprocess_time(self, time_str):
        """Convert time ke format dd/mm/yyyy"""
        try:
            if pd.isna(time_str):
                return None
            formats = ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%Y-%m-%d %H:%M:%S']
            for fmt in formats:
                try:
                    dt = datetime.strptime(str(time_str), fmt)
                    return dt.strftime('%d/%m/%Y')
                except:
                    continue
            return str(time_str)
        except:
            return None

    def preprocess_rating(self, rating):
        """Extract integer dari rating"""
        try:
            if pd.isna(rating):
                return None
            numbers = re.findall(r'\d+', str(rating))
            if numbers:
                return int(numbers[0])
            return None
        except:
            return None