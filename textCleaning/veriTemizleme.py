# Metinerdeki Fazla Boşlukları Ortadan Kaldi4
text = "Hello,             World!         2035"

#text.split()

cleanedText1 = " ".join(text.split())
print(cleanedText1)

# Büyükten Küçük Harf Çevirme
text = "Hello, World! 2035"
text.lower()  # küçük harfe cevir
cleanedText2 = text.lower()
print(cleanedText2)

# Noktalama İşaretlerini Kaldır
import string # tüm noktalama işaretlerini barındıran bir sabit bulundurcak bize
text = "Hello, World! 2035"
cleanedText3 = text.translate(str.maketrans("", "", string.punctuation))
print(cleanedText3)

# Ozel Karakterleri Kaldır
import re #Regular Expression= düzenli ifadelerle çalışmamızı sağlar
text = "Hello, World! 2035½"
cleanedText4 = re.sub(r"[^A-Za-z0-9\s]", "" , text) 
print(cleanedText4)

# Yazım hatalarını Düzelt
from textblob import TextBlob # textblob ile yazım hatasını düzeelcez

text = "hellio, wirld! 2035"
cleanedText5 = TextBlob(text).correct() # correct yazım hatalarını düzeltir
print(cleanedText5)

# HTML ve URL Etiketlerini Kaldır
from bs4 import BeautifulSoup
html_text = "<div>Hello, World! 2035 </div>"
cleanedText6 = BeautifulSoup(html_text, "html.parser").get_text() # get.text ile sadece metün kısmını alıyoruz beautiful soup ile parse ediyoruz
print(cleanedText6)





