'''custom_tokenizer.py
- preprocessing data, preserve some word, cut some stop word, clean unrrelated symbol, and preserve a ' - '. it's a sepperate token between message
- tokenize word with attacut
- have to avoid counting token with ' - ' '''

# defines a custom vectorizer class.
from pythainlp.corpus import thai_stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.pipeline import Pipeline

# include file for custom tf-idf
import scipy.sparse as sp
from sklearn.utils.validation import check_is_fitted, check_array, FLOAT_DTYPES, check_scalar
from sklearn.utils import _IS_32BIT
from sklearn.utils.fixes import _astype_copy_false

#import for reduce()
import functools


class CustomVectorizer(CountVectorizer): 
    
    # overwrite the build_analyzer method, and word_ngrams count
    stopwords = list(thai_stopwords())

    def _word_ngrams(self, tokens, stop_words=None):
            """Turn tokens into a sequence of n-grams after stop words filtering"""
            # handle stop words
            if stop_words is not None:
                tokens = [w for w in tokens if w not in stop_words]
            # handle token n-grams
            min_n, max_n = self.ngram_range
            if max_n != 1:
                original_tokens = tokens
                if min_n == 1:
                    # no need to do any slicing for unigrams
                    # just iterate through the original tokens
                    tokens = list(original_tokens)
                    min_n += 1
                else:
                    tokens = []

                n_original_tokens = len(original_tokens)

                # bind method outside of loop to reduce overhead
                tokens_append = tokens.append
                space_join = " ".join

                for n in range(min_n, min(max_n + 1, n_original_tokens + 1)):
                    for i in range(n_original_tokens - n + 1):
                        tokens_append(space_join(original_tokens[i : i + n]))
                tokens = [w for w in tokens if '-' not in w]
                # print('tokens in loop ',min_n, tokens)
            return tokens

    # create a custom analyzer for the vectorizer
    def build_analyzer(self):

        # load stop words using CountVectorizer's built in method
        # stop_words = self.get_stop_words()
        
        # create the analyzer that will be returned by this method
        def analyser(doc):
              
              # apply the preprocessing and tokenzation steps
              preprocressing_n_tokenizer = doc.split('-')
              preprocressing_n_tokenizer = [i.split(',') for i in preprocressing_n_tokenizer]
              result = functools.reduce(lambda a,b:a+['-']+b,preprocressing_n_tokenizer)

              # remove token that try to count cross message by containing '-' in their token
              # print('preprocressing_n_tokenizer ', preprocressing_n_tokenizer)
              # remove_separate_token = [word  for word in preprocressing_n_tokenizer if '-' not in word]
              # print('preprocressing_n_tokenizer ', preprocressing_n_tokenizer)
              # use CountVectorizer's _word_ngrams built in method
              # to remove stop words and extract n-grams
              return(self._word_ngrams(result, self.stopwords))
        return(analyser)

def _document_frequency(X):
    """Count the number of non-zero values for each feature in sparse X."""
    if sp.isspmatrix_csr(X):
        return np.bincount(X.indices, minlength=X.shape[1])
    else:
        return np.diff(X.indptr)

class CustomTfidfTranformer(TfidfTransformer): 

  def fit(self, X, y= None):
    """Learn the idf vector (global term weights).
        Parameters
        ----------
        X : sparse matrix of shape n_samples, n_features)
            A matrix of term/token counts.
        y : None
            This parameter is not needed to compute tf-idf.
        Returns
        -------
        self : object
            Fitted transformer.
        """
    # large sparse data is not supported for 32bit platforms because
    # _document_frequency uses np.bincount which works on arrays of
    # dtype NPY_INTP which is int32 for 32bit platforms. See #20923
    X = self._validate_data(X, accept_sparse=("csr", "csc"), accept_large_sparse=not _IS_32BIT)
    if not sp.issparse(X):
      X = sp.csr_matrix(X)
    dtype = X.dtype if X.dtype in FLOAT_DTYPES else np.float64
    if self.use_idf:
      n_samples, n_features = X.shape
      df = _document_frequency(X)
      df = df.astype(dtype, ** _astype_copy_false(df))
      # perform idf smoothing if required
      df += int(self.smooth_idf)
      n_samples += int(self.smooth_idf)
      # log+1 instead of log makes sure terms with zero idf don't get
      # suppressed entirely.
      idf = np.log(n_samples / df)
      self._idf_diag = sp.diags(idf,offsets=0,shape=(n_features, n_features),format="csr",dtype=dtype)
    return self


class CustomTfidfVectorizer(TfidfVectorizer): 
    def __init__(
        self,
        *,
        input="content",
        encoding="utf-8",
        decode_error="strict",
        strip_accents=None,
        lowercase=True,
        preprocessor=None,
        tokenizer=None,
        analyzer="word",
        stop_words=None,
        token_pattern=r"(?u)\b\w\w+\b",
        ngram_range=(1, 1),
        max_df=1.0,
        min_df=1,
        max_features=None,
        vocabulary=None,
        binary=False,
        dtype=np.float64,
        norm="l2",
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=False,
    ):

        super().__init__(
            input=input,
            encoding=encoding,
            decode_error=decode_error,
            strip_accents=strip_accents,
            lowercase=lowercase,
            preprocessor=preprocessor,
            tokenizer=tokenizer,
            analyzer=analyzer,
            stop_words=stop_words,
            token_pattern=token_pattern,
            ngram_range=ngram_range,
            max_df=max_df,
            min_df=min_df,
            max_features=max_features,
            vocabulary=vocabulary,
            binary=binary,
            dtype=dtype,
        )

        self._tfidf = CustomTfidfTranformer(
            norm=norm, use_idf=use_idf, smooth_idf=smooth_idf, sublinear_tf=sublinear_tf
        )
    # overwrite the build_analyzer method, and word_ngrams count
    stopwords = list(thai_stopwords())
    def _word_ngrams(self, tokens, stop_words=None):
            """Turn tokens into a sequence of n-grams after stop words filtering"""
            # handle stop words
            if stop_words is not None:
                tokens = [w for w in tokens if w not in stop_words]
            # handle token n-grams
            min_n, max_n = self.ngram_range
            if max_n != 1:
                original_tokens = tokens
                if min_n == 1:
                    # no need to do any slicing for unigrams
                    # just iterate through the original tokens
                    tokens = list(original_tokens)
                    min_n += 1
                else:
                    tokens = []

                n_original_tokens = len(original_tokens)

                # bind method outside of loop to reduce overhead
                tokens_append = tokens.append
                space_join = " ".join

            #     for n in range(min_n, min(max_n + 1, n_original_tokens + 1)):
            #         for i in range(n_original_tokens - n + 1):
            #             if '-' not in space_join(original_tokens[i : i + n]):
            #                 tokens_append(space_join(original_tokens[i : i + n]))
            # tokens = list(filter(lambda a: a != '-', tokens))
                for n in range(min_n, min(max_n + 1, n_original_tokens + 1)):
                        for i in range(n_original_tokens - n + 1):
                            # print(original_tokens[i : i + n])
                            tokens_append(space_join(original_tokens[i : i + n]))
                tokens = [w for w in tokens if '-' not in w]
            return tokens

    # create a custom analyzer for the vectorizer
    def build_analyzer(self):

        # load stop words using CountVectorizer's built in method
        # stop_words = self.get_stop_words()
        
        # create the analyzer that will be returned by this method
        def analyser(doc):
            
            # apply the preprocessing and tokenzation steps
            preprocressing_n_tokenizer = customize_text_tokenizer(doc)
            # remove token that try to count cross message by containing '-' in their token
            # print('preprocressing_n_tokenizer ', preprocressing_n_tokenizer)
            # remove_separate_token = [word  for word in preprocressing_n_tokenizer if '-' not in word]
            # print('preprocressing_n_tokenizer ', preprocressing_n_tokenizer)
            # use CountVectorizer's _word_ngrams built in method
            # to remove stop words and extract n-grams
            return(self._word_ngrams(preprocressing_n_tokenizer, self.stopwords))
        return(analyser)

class CustomTfidForTokenizedfVectorizer(CustomTfidfVectorizer):
    def build_analyzer(self):

          # load stop words using CountVectorizer's built in method
          # stop_words = self.get_stop_words()
          
          # create the analyzer that will be returned by this method
          def analyser(doc):
              
              # apply the preprocessing and tokenzation steps
              preprocressing_n_tokenizer = doc.split('-')
              preprocressing_n_tokenizer = [i.split(',') for i in preprocressing_n_tokenizer]
              result = functools.reduce(lambda a,b:a+['-']+b,preprocressing_n_tokenizer)

              # remove token that try to count cross message by containing '-' in their token
              # print('preprocressing_n_tokenizer ', preprocressing_n_tokenizer)
              # remove_separate_token = [word  for word in preprocressing_n_tokenizer if '-' not in word]
              # print('preprocressing_n_tokenizer ', preprocressing_n_tokenizer)
              # use CountVectorizer's _word_ngrams built in method
              # to remove stop words and extract n-grams
              return(self._word_ngrams(result, self.stopwords))
          return(analyser)

class CustomTokenizedVectorizerForOriginalTfidf(TfidfVectorizer): 
    def __init__(
        self,
        *,
        input="content",
        encoding="utf-8",
        decode_error="strict",
        strip_accents=None,
        lowercase=True,
        preprocessor=None,
        tokenizer=None,
        analyzer="word",
        stop_words=None,
        token_pattern=r"(?u)\b\w\w+\b",
        ngram_range=(1, 1),
        max_df=1.0,
        min_df=1,
        max_features=None,
        vocabulary=None,
        binary=False,
        dtype=np.float64,
        norm="l2",
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=False,
    ):

        super().__init__(
            input=input,
            encoding=encoding,
            decode_error=decode_error,
            strip_accents=strip_accents,
            lowercase=lowercase,
            preprocessor=preprocessor,
            tokenizer=tokenizer,
            analyzer=analyzer,
            stop_words=stop_words,
            token_pattern=token_pattern,
            ngram_range=ngram_range,
            max_df=max_df,
            min_df=min_df,
            max_features=max_features,
            vocabulary=vocabulary,
            binary=binary,
            dtype=dtype,
        )

        self._tfidf = TfidfTransformer(
            norm=norm, use_idf=use_idf, smooth_idf=smooth_idf, sublinear_tf=sublinear_tf
        )
    # overwrite the build_analyzer method, and word_ngrams count
    stopwords = list(thai_stopwords())
    def _word_ngrams(self, tokens, stop_words=None):
            """Turn tokens into a sequence of n-grams after stop words filtering"""
            # handle stop words
            if stop_words is not None:
                tokens = [w for w in tokens if w not in stop_words]
            # handle token n-grams
            min_n, max_n = self.ngram_range
            if max_n != 1:
                original_tokens = tokens
                if min_n == 1:
                    # no need to do any slicing for unigrams
                    # just iterate through the original tokens
                    tokens = list(original_tokens)
                    min_n += 1
                else:
                    tokens = []

                n_original_tokens = len(original_tokens)

                # bind method outside of loop to reduce overhead
                tokens_append = tokens.append
                space_join = " ".join

            #     for n in range(min_n, min(max_n + 1, n_original_tokens + 1)):
            #         for i in range(n_original_tokens - n + 1):
            #             if '-' not in space_join(original_tokens[i : i + n]):
            #                 tokens_append(space_join(original_tokens[i : i + n]))
            # tokens = list(filter(lambda a: a != '-', tokens))
                for n in range(min_n, min(max_n + 1, n_original_tokens + 1)):
                        for i in range(n_original_tokens - n + 1):
                            # print(original_tokens[i : i + n])
                            tokens_append(space_join(original_tokens[i : i + n]))
                tokens = [w for w in tokens if '-' not in w]
            return tokens

    # create a custom analyzer for the vectorizer
    def build_analyzer(self):

        # load stop words using CountVectorizer's built in method
        # stop_words = self.get_stop_words()
        
        # create the analyzer that will be returned by this method
        def analyser(doc):
              
              # apply the preprocessing and tokenzation steps
              preprocressing_n_tokenizer = doc.split('-')
              preprocressing_n_tokenizer = [i.split(',') for i in preprocressing_n_tokenizer]
              result = functools.reduce(lambda a,b:a+['-']+b,preprocressing_n_tokenizer)

              # remove token that try to count cross message by containing '-' in their token
              # print('preprocressing_n_tokenizer ', preprocressing_n_tokenizer)
              # remove_separate_token = [word  for word in preprocressing_n_tokenizer if '-' not in word]
              # print('preprocressing_n_tokenizer ', preprocressing_n_tokenizer)
              # use CountVectorizer's _word_ngrams built in method
              # to remove stop words and extract n-grams
              return(self._word_ngrams(result, self.stopwords))
        return(analyser)