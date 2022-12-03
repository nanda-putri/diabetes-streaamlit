import streamlit as st
import numpy as np
import pandas as pd
from streamlit_option_menu import option_menu



with st.sidebar:
    st.title("Diabtes Melitus")
    st.sidebar.image("pict.jpg")

choose = option_menu("Diabetic Detection", ["Home", "Diagnosis", "About Me"], 
    icons=['house', "list-task", 'person lines fill'], 
    menu_icon='cast', default_index=0, orientation="horizontal",
    styles={
        "container":{"background-color":"#2F4F44"},
        "nav-link":{"--hover-color":"#8FBC88"},
        "nav-link-selected":{"background-color":"#8FBC88"}
    })


if choose == "Home":
    st.title("Apa itu Diabetes Melitus???")
    st.write("Diabetes melitus atau penyakit kencing manis adalah suatu penyakit kronis ketika kadar gula darah (glukosa) di dalam tubuh terlampau tinggi dan berada di atas normal. Tingginya kadar gula darah penyebab diabetes melitus dapat terjadi karena kurangnya hormon insulin ataupun tidak tercukupinya hormon insulin karena tubuh tidak dapat menggunakannya secara optimal (resistensi insulin). Kedua hal tersebut dapat terjadi secara tunggal atau kombinasi.")
    st.write("Glukosa sendiri berasal dari sumber makanan yang dikonsumsi lalu diolah tubuh dan menjadi sumber energi utama bagi sel tubuh manusia. Kadar gula darah dikendalikan oleh hormon insulin yang diproduksi oleh pankreas. Pankreas melepaskan insulin ini ke dalam aliran darah dan membantu glukosa (zat gula) dari makanan masuk ke dalam sel-sel seluruh tubuh. Akan tetapi jika tubuh tidak membuat cukup insulin atau insulin tidak bekerja dengan baik dapat menyebabkan glukosa tidak bisa masuk ke dalam sel dan membuat glukosa menumpuk dalam darah. Hal ini yang membuat kadar gula dalam darah menjadi tinggi dan menyebabkan terjadinya penyakit diabetes melitus.Kadar glukosa darah yang tinggi dapat menimbulkan gangguan pada organ tubuh, termasuk merusak pembuluh darah kecil di organ ginjal, jantung, mata, ataupun sistem saraf. Ketika tidak ditangani dengan baik pada akhirnya dapat menyebabkan terjadinya komplikasi penyakit seperti jantung, stroke, penyakit ginjal, kebutaan, dan kerusakan pada saraf.")
    st.write("Gejala diabetes akibat kadar glukosa darah yang tinggi dapat meliputi rasa haus yang meningkat (polidipsia), peningkatan buang air kecil (poliuria), penglihatan kabur, mudah mengantuk, mual, menurunnya daya tahan tubuh, dan meningkatnya rasa lapar (polifagia).")
    st.write("International Diabetes Federation (IDF) memperkirakan sekitar 463 juta orang berusia 20-79 tahun di seluruh dunia mengidap diabetes pada 2019. Angka ini setara dengan 9,3% dari total populasi dunia dan diprediksikan naik menjadi 700 juta pada tahun 2045. Indonesia sendiri menempati peringkat ke-7 sebagai negara dengan jumlah pengidap diabetes tertinggi di dunia, yaitu sebesar 10,7 juta.")


elif choose == "Diagnosis":
    st.title("Diabetic Detection & Classification")
    pilih= option_menu(None, ["Datasets", "Preprocessing", "Modeling","Implementation"], 
        # icons=['house', "list-task", 'person lines fill'], 
        menu_icon='app-indicator', default_index=0, orientation="horizontal",
        styles={
            "container":{"font-size": "15px","background-color":"#000"},
            # "icon":{"font-size": "14px"},
            "nav-link":{"--hover-color":"#8FBC88"},
            "nav-link-selected":{"font-size": "12px","background-color":"#8FBC88"}
        })

    if pilih == "Datasets":
        st.title("Datasets")
        st.write("Untuk mendukung proses klasifikasi website ini, saya menggunakan datasets publik yang diambil dari kaggle. Datasets ini ada 9 kolom yang terdiri dari 8 attribute dan 1 class. Atributnya antara lain Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, dan Age. Sedangkan classnya adalah Outcome. Ini merupakan link akses untuk datasets yang saya gunakan https://www.kaggle.com/datasets/mathchi/diabetes-data-set.")
        df = pd.read_csv("https://raw.githubusercontent.com/nanda-putri/datamining/gh-pages/diabetes.csv")
        st.dataframe(df)
    
    elif pilih == "Preprocessing":
        st.title("Preprocessing")
        # st.write("Datasets dilakukan preprocessing sebelum di uji dengan metode terpilih.")

        url = "https://raw.githubusercontent.com/nanda-putri/datamining/gh-pages/diabetes.csv"
        data = pd.read_csv(url)

        # data asli
        st.subheader("Data Asli")
        data

        # preprocessing
        data = data [["Pregnancies","Glucose", "BloodPressure", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]]

        x = data.drop(columns = ["Outcome"])
        #y = data.Outcome
        
        st.subheader("Data Preprocessing")

        from sklearn.preprocessing import MinMaxScaler

        scaler = MinMaxScaler()
        #scaler.fit(features)
        #scaler.transform(features)
        scaled = scaler.fit_transform(x)
        features_names = x.columns.copy()
        #features_names.remove('label')
        scaled_features = pd.DataFrame(scaled, columns=features_names)
        scaled_features

    elif pilih == "Modeling":
        st.title("Modeling")

        # load datasets
        url = "https://raw.githubusercontent.com/nanda-putri/datamining/gh-pages/diabetes.csv"
        data = pd.read_csv(url)

        # seleksi fitur
        data = data [["Pregnancies","Glucose", "BloodPressure", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]]

        # drop class
        x = data.drop(columns = ["Outcome"])
        y = data.Outcome
        # x

        # normalisasi data
        from sklearn.preprocessing import MinMaxScaler

        scaler = MinMaxScaler()
        #scaler.fit(features)
        #scaler.transform(features)
        scaled = scaler.fit_transform(x)
        features_names = x.columns.copy()
        #features_names.remove('label')
        scaled_features = pd.DataFrame(scaled, columns=features_names)
        # scaled_features.head(30)

        # save model preprocessing
        import joblib
        filename = "norm.sav"
        joblib.dump(scaler,filename)

        y = data['Outcome'].values

        # split datasets
        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1) 

        # cek ukuran data
        st.subheader("Jumlah Data")
        st.write("Jumlah Total Datasets = ",x.shape)
        st.write("Total X train = ",x_train.shape) 
        st.write("Total y train = ",y_train.shape)
        st.write("Total x test = ", x_test.shape)
        st.write("Total y test = ", y_test.shape)

        # KNN
        st.header("KNN")
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import GridSearchCV
        from sklearn.metrics import classification_report

        # my_param_grid = {'n_neighbors':[2,3,5,7], 'weights': ['distance', 'uniform']}
        # GridSearchCV(estimator=KNeighborsClassifier(), param_grid=my_param_grid, refit=True, verbose=3, cv=3)
        # knn = GridSearchCV(KNeighborsClassifier(), param_grid=my_param_grid, refit=True, verbose=3, cv=3)

        knn = KNeighborsClassifier(n_neighbors = 3)
        knn.fit(x_train, y_train)
        knn.score(x_test,y_test)

        knn.fit(x_train, y_train)

        # save moodel KNN
        filenameModelKnnNorm = 'modelKNN.pkl'
        joblib.dump(knn, filenameModelKnnNorm)

        # predict
        predTestKNN = knn.predict(x_test)
        # predTestKNN

        # st.write(round((knn.score(x_train,y_train)* 100),2))
        st.write(round((knn.score(x_test, y_test)*100),2))
        accuracy_score(y_test, predTestKNN)
        predTestKNN = knn.predict([[4,110,92,0,37.6,0.191,30]])
        predTestKNN

        # GaussianNB
        st.header("GaussianNB")
        from sklearn.naive_bayes import GaussianNB

        gaussian = GaussianNB()
        gaussian.fit(x_train, y_train)

        # save model GNB
        filenameModelGNB = 'ModelGNB.pkl'
        joblib.dump(gaussian, filenameModelGNB)
        predTestGNB = gaussian.predict(x_test)



        # predict
        predTestGNB = gaussian.predict(x_test)
        predTestGNB = gaussian.predict(x_test)
        accuracy_score(y_test,predTestGNB)
        # st.write(round((gaussian.score(x_train,y_train)*100),2))
        st.write(round((gaussian.score(x_test,y_test)*100),2))
        predTestGNB = gaussian.predict([[4,110,92,0,37.6,0.191,30]])
        predTestGNB

        # Decision Tree
        st.header("Decision Tree")
        from sklearn.tree import DecisionTreeClassifier

        tr = DecisionTreeClassifier()
        tr = tr.fit(x,y)

        # save model Decision Tree
        filenameModelDT = 'modelDT.pkl'
        joblib.dump(tr, filenameModelDT)

        # predict
        predTestDT = tr.predict(x_test)
        accuracy_score(y_test,predTestDT)
        # tr.score(x_train,y_train)*100
        tr.score(x_test,y_test)*100
        predTestDT = tr.predict([[4,110,92,0,37.6,0.191,30]])
        predTestDT


    elif pilih == "Implementation":
        st.title("Implementation")
        st.markdown("Check your Diabetes Here!!!")
        st.write("6,148,72,0,33.6,0.627,50")
        #input data
        # pregnancies
        pregnan = st.number_input("Berapa kali hamil : ")
        # glukosa
        glukosa = st.number_input("Glukosa (2 jam terakhir) : ")

        #tekanan darah
        tekanan = st.number_input("Tekanan Darah (mm Hg) : ")

        #insulin
        insulin = st.number_input("Insulin (mu U/ml) : ",)

        #BMI
        bmi = st.number_input("BMI (weight in kg/(height in m)^2) : ")

        #resiko
        resiko = st.number_input("Resiko Keturunan Diabetes : ")

        #umur
        umur = st.number_input("Age (years) : ")

        #button 
        # ok = st.button("Check")

        # data
        hitung = [pregnan,glukosa, tekanan, insulin, bmi, resiko, umur]

        # perhitungan
        # load datasets
        url = "https://raw.githubusercontent.com/nanda-putri/datamining/gh-pages/diabetes.csv"
        data = pd.read_csv(url)

        # seleksi fitur
        data = data [["Pregnancies","Glucose", "BloodPressure", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]]


        # normalisasi data
        from sklearn.preprocessing import MinMaxScaler

        x = data.drop(columns = ["Outcome"])

        scaler = MinMaxScaler()
        #scaler.fit(features)
        #scaler.transform(features)
        scaled = scaler.fit_transform(x)
        features_names = x.columns.copy()
        #features_names.remove('label')
        scaled_features = pd.DataFrame(scaled, columns=features_names)
        # scaled_features.head(30)

        # save model preprocessing
        import joblib
        filename = "norm.sav"
        joblib.dump(scaler,filename)

        y = data['Outcome'].values

        # split datasets
        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1) 

        # normalisasi data input
        data_min = x.min()
        data_max = x.max()

        normInput = ((hitung-data_min)/(data_max-data_min))

        # check

        #  pilih model
        pilih = st.radio("Pilih Model : ",("K-NearestNeighbhor", "Gaussian Naive Baiyes", "Decision Tree"))
        # KNN
        if pilih == "K-NearestNeighbhor":
            ok = st.button("Check")
            if ok:
                from sklearn.neighbors import KNeighborsClassifier
                from sklearn.metrics import accuracy_score
                from sklearn.model_selection import GridSearchCV
                from sklearn.metrics import classification_report
                knn = KNeighborsClassifier(n_neighbors = 3)
                knn.fit(x_train, y_train)
                knn.score(x_test,y_test)

                knn.fit(x_train, y_train)

                # save moodel KNN
                filenameModelKnnNorm = 'modelKNN.pkl'
                joblib.dump(knn, filenameModelKnnNorm)

                # predict
                predTestKNN = knn.predict(x_test)
                #predTestKNN[0]

                if (predTestKNN[0] == 0):
                    st.write("Negative Diabetes")
                elif  (predTestKNN[0] == 1):
                    st.write("Positive Diabetes")

        # GaussianNB
        elif pilih == "Gaussian Naive Baiyes":
            ok = st.button("Check")
            if ok:
                from sklearn.naive_bayes import GaussianNB

                gaussian = GaussianNB()
                gaussian.fit(x_train, y_train)

                # save model GNB
                filenameModelGNB = 'ModelGNB.pkl'
                joblib.dump(gaussian, filenameModelGNB)
                predTestGNB = gaussian.predict(x_test)
                if (predTestGNB[0] == 0):
                    st.write("Negative Diabetes")
                elif  (predTestGNB[0] == 1):
                    st.write("Positive Diabetes")

        elif pilih == "Decision Tree":
            ok = st.button("Check")
            if ok:
                from sklearn.tree import DecisionTreeClassifier
                tr = DecisionTreeClassifier()
                tr = tr.fit(x,y)

                # save model Decision Tree
                filenameModelDT = 'modelDT.pkl'
                joblib.dump(tr, filenameModelDT)

                # predict
                predTestDT = tr.predict(x_test)
                if (predTestDT[0] == 0):
                    st.write("Negative Diabetes")
                elif  (predTestDT[0] == 1):
                    st.write("Positive Diabetes")

elif choose == "About Me":
    "Ini adalah about me"
