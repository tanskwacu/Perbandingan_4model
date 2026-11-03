from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

st.set_page_config(page_title='CarLab · Analisis & Perbandingan Model', page_icon='🚘', layout='wide')
ROOT = Path(__file__).resolve().parent
NUM = ['wheelbase','curbweight','enginesize','boreratio','horsepower','citympg','highwaympg','carlength','carwidth']
CAT = ['enginetype','fueltype','aspiration','carbody','cylindernumber','drivewheel']
LABEL = {'wheelbase':'Jarak sumbu roda','curbweight':'Berat mobil','enginesize':'Ukuran mesin','boreratio':'Rasio bore','horsepower':'Tenaga mesin','citympg':'Efisiensi kota (mpg)','highwaympg':'Efisiensi jalan raya (mpg)','carlength':'Panjang mobil','carwidth':'Lebar mobil','enginetype':'Jenis mesin','fueltype':'Bahan bakar','aspiration':'Aspiration / turbo','carbody':'Jenis bodi','cylindernumber':'Jumlah silinder','drivewheel':'Penggerak'}
PAGES = ['Ringkasan','Eksplorasi Data','Visualisasi','Persiapan Data','Perbandingan Model','Prediksi Harga','Kesimpulan']
st.markdown('''<style>.block-container{padding-top:2rem;max-width:1400px}div[data-testid="stMetric"]{border:1px solid #48536644;border-radius:14px;padding:16px}h1{letter-spacing:-1px} .eyebrow{color:#55b7cb;font-size:13px;letter-spacing:3px;font-weight:700} </style>''', unsafe_allow_html=True)

@st.cache_data
def load_data(path, stamp):
    return pd.read_csv(path)

@st.cache_resource
def fit_models(df):
    clean = df.dropna(subset=['price']).drop_duplicates().copy()
    X, y = clean[NUM + CAT], clean.price
    train, test = train_test_split(clean.index, test_size=.2, random_state=100)
    models, scores, predictions, messages = {}, [], [], []
    for name, estimator in [('Linear',LinearRegression()),('Polynomial',LinearRegression()),('Ridge',Ridge(alpha=1)),('Lasso',Lasso(alpha=1,max_iter=100000))]:
        prep = ColumnTransformer([
            ('numerik',Pipeline([('imputer',SimpleImputer(strategy='median')),('skala',MinMaxScaler())]),NUM),
            ('kategori',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('dummy',OneHotEncoder(drop='first',handle_unknown='ignore',sparse_output=False))]),CAT)
        ])
        steps = [('persiapan',prep)]
        if name == 'Polynomial':
            steps.append(('polynomial',PolynomialFeatures(degree=2,include_bias=False)))
        steps.append(('regresi',estimator))
        model = Pipeline(steps)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            model.fit(X.loc[train],y.loc[train])
            for split, ix in [('Latih',train),('Uji',test)]:
                pred = model.predict(X.loc[ix])
                scores.append({'Model':name,'Data':split,'R²':r2_score(y.loc[ix],pred),'RMSE':np.sqrt(mean_squared_error(y.loc[ix],pred)),'MAE':mean_absolute_error(y.loc[ix],pred)})
                if split == 'Uji':
                    for idx, actual, value in zip(ix,y.loc[ix],pred):
                        predictions.append({'Model':name,'ID':int(clean.loc[idx,'car_ID']),'Mobil':clean.loc[idx,'CarName'],'Aktual':actual,'Prediksi':value,'Residual':actual-value})
            messages.extend([f'{name}: {w.message}' for w in caught])
        models[name] = model
    return models,pd.DataFrame(scores),pd.DataFrame(predictions),train,test,list(dict.fromkeys(messages)),clean

path = ROOT / 'CarPrice.csv'
if not path.exists():
    st.error('CarPrice.csv harus berada di folder yang sama dengan app.py.'); st.stop()
df = load_data(str(path),path.stat().st_mtime_ns)
with st.spinner('Menyiapkan empat model dan hasil evaluasi…'):
    models,scores,pred,train_idx,test_idx,messages,clean = fit_models(df)
rank = scores[scores.Data == 'Uji'].sort_values('RMSE').reset_index(drop=True)
best = rank.iloc[0]
st.sidebar.markdown('## 🚘 CarLab')
st.sidebar.caption('ANALISIS HARGA MOBIL')
page = st.sidebar.radio('Navigasi',PAGES)
st.sidebar.divider()
st.sidebar.caption(f'{len(df)} mobil · 4 model\n\n80% latihan / 20% pengujian\n\nRandom state: 100')
st.sidebar.info('Harga mengikuti satuan dataset. Sumber asli, tahun, serta mata uang belum terverifikasi; bukan harga pasar terkini.')
st.markdown('<div class="eyebrow">CARLAB / STUDI REGRESI</div>',unsafe_allow_html=True)
st.title(page)

def table(data):
    st.dataframe(data,width='stretch',hide_index=True)

def chart(data):
    st.altair_chart(data,width='stretch')

def download(data,name,label='Unduh tabel CSV'):
    st.download_button(label,data.to_csv(index=False).encode('utf-8-sig'),name,'text/csv')

def corr_chart(data):
    c = data.corr().rename_axis('Fitur A').reset_index().melt('Fitur A',var_name='Fitur B',value_name='Korelasi')
    chart(alt.Chart(c).mark_rect().encode(x='Fitur A:N',y='Fitur B:N',color=alt.Color('Korelasi:Q',scale=alt.Scale(domain=[-1,1],scheme='redblue')),tooltip=['Fitur A','Fitur B',alt.Tooltip('Korelasi:Q',format='.3f')]).properties(height=520))

if page == 'Ringkasan':
    st.write('Membandingkan Regresi Linear, Polynomial, Ridge, dan Lasso untuk memperkirakan harga mobil dari spesifikasinya.')
    for col,label,value in zip(st.columns(4),['Jumlah mobil','Kolom asli','Harga rata-rata','Rentang harga'],[len(df),len(df.columns),f'{df.price.mean():,.0f}',f'{df.price.min():,.0f} – {df.price.max():,.0f}']):
        col.metric(label,value)
    st.subheader('Hasil utama pengujian')
    st.success(f"{best.Model} memiliki RMSE uji terendah: {best.RMSE:,.2f}, dengan R² {best['R²']:.3f} pada {len(test_idx)} mobil uji.")
    chart(alt.Chart(rank).mark_bar(cornerRadiusEnd=5).encode(x=alt.X('RMSE:Q',title='RMSE uji — lebih kecil lebih baik'),y=alt.Y('Model:N',sort='x'),color=alt.Color('Model:N',legend=None),tooltip=['Model','RMSE','R²']).properties(height=240))
    st.caption('Peringkat hanya berlaku untuk pembagian data dan pengaturan ini. Data uji tidak digunakan untuk melatih model atau penyesuaian skala.')
    st.subheader('Alur penelitian')
    st.write('Data → pemeriksaan kualitas → eksplorasi → pembagian latih/uji → persiapan fitur → pelatihan empat model → evaluasi → kesimpulan.')
    st.info('Mulai dari Eksplorasi Data untuk memahami dataset, atau buka Perbandingan Model untuk melihat hasil inti proyek.')

elif page == 'Eksplorasi Data':
    st.write('Pemeriksaan berikut menggunakan dataset asli sebelum persiapan model.')
    a,b,c = st.columns(3)
    a.metric('Sel kosong',int(df.isna().sum().sum())); b.metric('Baris duplikat',int(df.duplicated().sum())); c.metric('Nama mobil unik',df.CarName.nunique())
    t1,t2,t3,t4 = st.tabs(['Dataset','Struktur & kualitas','Statistik','Kamus data'])
    with t1:
        q=st.text_input('Cari nama mobil')
        body=st.selectbox('Filter bodi',['Semua']+sorted(df.carbody.unique()))
        shown=df[df.CarName.str.contains(q,case=False,regex=False,na=False)]
        if body!='Semua': shown=shown[shown.carbody==body]
        st.caption(f'{len(shown)} baris ditampilkan. Filter hanya berlaku pada tabel ini.')
        table(shown); download(shown,'data_mobil.csv')
    with t2:
        table(pd.DataFrame({'Kolom':df.columns,'Tipe':df.dtypes.astype(str).values,'Terisi':df.notna().sum().values,'Kosong':df.isna().sum().values,'Nilai unik':df.nunique().values}))
        st.subheader('Baris duplikat identik'); table(df[df.duplicated(keep=False)])
        st.caption('Duplikat dihitung atas seluruh kolom, termasuk car_ID. Nama sama tidak otomatis berarti data duplikat.')
    with t3:
        st.subheader('Numerik'); table(df.describe().T.rename_axis('Kolom').reset_index())
        st.subheader('Kategori'); table(df.describe(include='object').T.rename_axis('Kolom').reset_index())
    with t4:
        meaning={'car_ID':'Identitas baris; bukan fitur prediksi','symboling':'Indeks symboling; definisi sumber belum terverifikasi','CarName':'Nama mobil; tidak sama dengan merek','doornumber':'Jumlah pintu','enginelocation':'Letak mesin','carheight':'Tinggi mobil','fuelsystem':'Sistem bahan bakar','stroke':'Langkah piston','compressionratio':'Rasio kompresi','peakrpm':'Putaran mesin puncak (rpm)','price':'Harga target; mata uang belum terverifikasi',**LABEL}
        table(pd.DataFrame([{'Kolom':c,'Arti':meaning.get(c,c),'Dipakai model':c in NUM+CAT} for c in df.columns]))
        st.warning('Satuan dimensi, berat, dan ukuran mesin tidak dicantumkan di file CSV. Nilai dipertahankan sesuai data; jangan menafsirkannya otomatis sebagai cm, kg, atau cc.')

elif page == 'Visualisasi':
    st.write('Eksplorasi seluruh data. Hubungan pada grafik menunjukkan asosiasi, bukan bukti sebab-akibat.')
    tab1,tab2,tab3,tab4=st.tabs(['Distribusi harga','Kategori & nama','Spesifikasi numerik','Korelasi'])
    with tab1:
        chart(alt.Chart(df).mark_bar(color='#58a6c8').encode(x=alt.X('price:Q',bin=alt.Bin(maxbins=25),title='Harga'),y=alt.Y('count():Q',title='Jumlah mobil'),tooltip=['count()']).properties(height=330))
        st.write(f'Median harga {df.price.median():,.2f}; rata-rata {df.price.mean():,.2f}. Histogram menunjukkan rentang harga yang paling banyak diwakili dataset.')
        chart(alt.Chart(df).mark_boxplot().encode(x=alt.X('price:Q',title='Harga')).properties(height=100))
        st.caption('Boxplot merangkum median, kuartil, dan nilai jauh dari sebaran utama; nilai jauh tidak otomatis salah.')
    with tab2:
        eda=df.copy()
        fixes={'maxda':'mazda','porcshce':'porsche','toyouta':'toyota','vokswagen':'volkswagen','vw':'volkswagen'}
        eda['Merek']=eda.CarName.str.lower().str.split().str[0].replace(fixes)
        cat=st.selectbox('Kategori yang ingin dianalisis',['CarName','Merek','cylindernumber','fueltype','carbody','enginetype','doornumber','aspiration','enginelocation','drivewheel','fuelsystem'])
        agg=eda.groupby(cat).price.agg(Jumlah='size',Rata_rata='mean',Median='median').reset_index()
        limit=st.slider('Jumlah kategori pada grafik',5,max(5,len(agg)),min(15,max(5,len(agg))))
        order=st.radio('Urutkan kategori menurut',['Jumlah','Rata_rata'],horizontal=True)
        shown=agg.sort_values(order,ascending=False).head(limit)
        left,right=st.columns(2)
        with left: chart(alt.Chart(shown).mark_bar().encode(x='Jumlah:Q',y=alt.Y(f'{cat}:N',sort='-x'),tooltip=list(shown.columns)).properties(height=380))
        with right: chart(alt.Chart(shown).mark_bar(color='#dca35c').encode(x=alt.X('Rata_rata:Q',title='Rata-rata harga'),y=alt.Y(f'{cat}:N',sort='-x'),tooltip=list(shown.columns)).properties(height=380))
        chart(alt.Chart(eda[eda[cat].isin(shown[cat])]).mark_boxplot().encode(x=f'{cat}:N',y='price:Q').properties(height=300))
        top=agg.sort_values('Rata_rata',ascending=False).iloc[0]
        st.write(f"Kategori dengan rata-rata harga tertinggi: {top[cat]} ({top.Rata_rata:,.2f}), dari {int(top.Jumlah)} mobil. Perhatikan ukuran kelompok sebelum membandingkan.")
        st.caption('Grafik menampilkan kategori teratas sesuai pilihan. Tabel di bawah tetap mencakup semua kategori. Merek diturunkan dari kata pertama nama dan koreksi ejaan umum; hanya untuk eksplorasi.')
        table(agg.sort_values(order,ascending=False)); download(agg,'ringkasan_kategori.csv')
    with tab3:
        numeric=[c for c in df.select_dtypes('number').columns if c not in ['car_ID','price']]
        features=st.multiselect('Spesifikasi yang ditampilkan',numeric,default=['enginesize','horsepower','curbweight','carwidth','citympg','highwaympg'])
        st.caption('Pilih juga carlength, carheight, boreratio, stroke, compressionratio, peakrpm, dan wheelbase untuk seluruh analisis notebook.')
        cols=st.columns(2)
        for i,c in enumerate(features):
            with cols[i%2]:
                chart(alt.Chart(df).mark_circle(size=45,opacity=.65).encode(x=alt.X(f'{c}:Q',scale=alt.Scale(zero=False)),y='price:Q',tooltip=['CarName',c,'price']).properties(height=250,title=f'{c} vs harga'))
                st.caption(f'Korelasi Pearson dengan harga: {df[c].corr(df.price):.3f}. Setiap titik mewakili satu mobil.')
    with tab4:
        numeric=df.select_dtypes('number').drop(columns=['car_ID'])
        corr_chart(numeric)
        st.caption('Mendekati +1: bergerak searah; mendekati −1: berlawanan; mendekati 0: hubungan linear lemah. ID dikeluarkan karena hanya penanda baris.')
        table(numeric.corr()['price'].drop('price').sort_values(key=abs,ascending=False).rename('Korelasi dengan harga').rename_axis('Fitur').reset_index())

elif page == 'Persiapan Data':
    st.subheader('Proses yang digunakan dashboard')
    st.markdown(f'''1. Periksa data; keluarkan duplikat identik dan baris tanpa harga target bila ada. **{len(clean)} baris** tersedia untuk model.
2. Gunakan **9 fitur numerik dan 6 fitur kategori** yang dipilih notebook, tanpa `Car_category`.
3. Bagi data **80:20** dengan `random_state=100`: **{len(train_idx)} latihan**, **{len(test_idx)} pengujian**.
4. Isi fitur numerik kosong dengan median latihan; kategori kosong dengan modus latihan.
5. Terapkan **MinMaxScaler** pada fitur numerik dan **one-hot encoding** pada kategori. Seluruh parameter dipelajari hanya dari data latihan.
6. Latih empat model. Polynomial menambahkan pangkat dan interaksi derajat 2 setelah encoding.
7. Evaluasi pada harga asli: target tidak dinormalisasi atau dibulatkan.''')
    table(pd.DataFrame([{'Fitur':c,'Jenis':'Numerik' if c in NUM else 'Kategori','Keterangan':LABEL[c]} for c in NUM+CAT]))
    st.subheader('Perbaikan dari notebook')
    table(pd.DataFrame([
        {'Bagian':'Car_category','Notebook':'Dibentuk dari rata-rata harga per CarName sebelum pembagian data','Dashboard':'Dihapus: harga target belum diketahui saat memprediksi mobil baru'},
        {'Bagian':'Normalisasi','Notebook':'dfdrop dinormalisasi setelah x_train/x_test dibuat; tidak konsisten dengan input model','Dashboard':'Scaler menjadi bagian pipeline dan dilatih hanya pada data latihan'},
        {'Bagian':'Encoding','Notebook':'Dummy dibuat sebelum pembagian','Dashboard':'Kategori dipelajari hanya dari data latihan; kategori baru ditangani tanpa crash'},
        {'Bagian':'Harga','Notebook':'Harga diubah ke integer','Dashboard':'Presisi harga asli dipertahankan'},
        {'Bagian':'Polynomial','Notebook':'fit_transform pada data uji','Dashboard':'Hanya transform data uji dengan pipeline hasil latihan'}]))
    st.caption('Fitur lain, termasuk nama mobil dan ID, tetap tampil dalam eksplorasi tetapi tidak menjadi masukan model. Pemilihan fitur mengikuti notebook, bukan hasil seleksi fitur baru.')
    pre=models['Linear'].named_steps['persiapan']
    encoded=pd.DataFrame(pre.transform(clean.loc[train_idx,NUM+CAT]),columns=pre.get_feature_names_out())
    with st.expander('Lihat hasil encoding dan normalisasi data latihan'):
        st.write(f'{encoded.shape[1]} fitur setelah encoding. Nilai data uji bisa di luar 0–1 jika melampaui rentang latihan.')
        table(encoded); download(encoded,'fitur_latihan.csv')
    with st.expander('Heatmap fitur hasil persiapan'):
        z=encoded.copy(); z['price']=clean.loc[train_idx,'price'].to_numpy(); corr_chart(z)
    with st.expander('Pembagian baris latihan dan pengujian'):
        part=clean[['car_ID','CarName','price']].copy(); part['Bagian']=np.where(part.index.isin(train_idx),'Latih','Uji'); table(part)
    st.subheader('Pengaturan model')
    table(pd.DataFrame({'Model':['Linear','Polynomial','Ridge','Lasso'],'Pengaturan':['Regresi linear, intercept aktif','Derajat 2, interaksi seluruh fitur encoded, regresi linear','alpha=1, intercept aktif','alpha=1, max_iter=100000, intercept aktif']}))
    st.info('Tidak ada pencarian hyperparameter. Polynomial dapat memiliki lebih banyak fitur daripada jumlah data latihan dan mudah overfit. Hasil tidak dipaksa agar sama dengan notebook.')

elif page == 'Perbandingan Model':
    st.write(f'Empat model, masukan yang sama, dan {len(test_idx)} baris uji yang sama. Peringkat ditentukan berdasarkan RMSE uji terendah.')
    st.caption('R² bukan persentase akurasi dan bisa negatif. RMSE menghukum kesalahan besar; MAE adalah rata-rata kesalahan absolut. RMSE dan MAE mengikuti satuan harga.')
    table(scores.pivot(index='Model',columns='Data',values=['R²','RMSE','MAE']).round(3).reset_index())
    download(scores,'evaluasi_empat_model.csv')
    a,b=st.columns(2)
    for col,metric in [(a,'R²'),(b,'RMSE')]:
        with col:
            st.subheader(f'{metric}: latihan vs pengujian')
            chart(alt.Chart(scores).mark_bar().encode(x='Model:N',xOffset='Data:N',y=f'{metric}:Q',color='Data:N',tooltip=['Model','Data',alt.Tooltip(f'{metric}:Q',format=',.3f')]).properties(height=300))
    st.subheader('Harga asli vs prediksi • data uji')
    bounds=[float(min(pred.Aktual.min(),pred.Prediksi.min())),float(max(pred.Aktual.max(),pred.Prediksi.max()))]
    line=pd.DataFrame({'Aktual':bounds,'Prediksi':bounds})
    cols=st.columns(2)
    for i,name in enumerate(models):
        with cols[i%2]:
            points=alt.Chart(pred[pred.Model==name]).mark_circle(size=55).encode(x=alt.X('Aktual:Q',scale=alt.Scale(domain=bounds)),y=alt.Y('Prediksi:Q',scale=alt.Scale(domain=bounds)),tooltip=['Mobil','Aktual','Prediksi','Residual'])
            ideal=alt.Chart(line).mark_line(color='#e49a58',strokeDash=[5,5]).encode(x='Aktual:Q',y='Prediksi:Q')
            chart((points+ideal).properties(title=name,height=260))
    st.caption('Skala sama untuk seluruh model. Titik dekat garis putus-putus berarti prediksi mendekati harga asli. Prediksi ekstrem tetap ditampilkan, tidak disembunyikan.')
    st.subheader('Pola kesalahan / residual')
    chart(alt.Chart(pred).mark_circle(opacity=.65).encode(x='Prediksi:Q',y='Residual:Q',color='Model:N',tooltip=['Model','Mobil','Aktual','Prediksi','Residual']).properties(height=330)+alt.Chart(pd.DataFrame({'nol':[0]})).mark_rule(color='#dca35c').encode(y='nol:Q'))
    st.caption('Residual = aktual − prediksi. Positif berarti model menaksir terlalu rendah; negatif berarti terlalu tinggi. Pola melengkung atau sebaran melebar menandakan kesalahan yang tidak merata.')
    chart(alt.Chart(pred).mark_boxplot().encode(x='Model:N',y='Residual:Q',color=alt.Color('Model:N',legend=None)).properties(height=260))
    st.subheader('Selisih performa latihan dan pengujian')
    gap=scores.pivot(index='Model',columns='Data',values='R²'); gap['Selisih R² (latih − uji)']=gap.Latih-gap.Uji
    table(gap.reset_index()); st.caption('Selisih besar dapat menunjukkan overfitting. Nilai ini merupakan petunjuk, bukan pengujian statistik.')
    st.subheader('Prediksi setiap mobil uji')
    wide=pred.pivot(index=['ID','Mobil','Aktual'],columns='Model',values='Prediksi').reset_index()
    table(wide); download(wide,'prediksi_data_uji.csv')
    if messages:
        with st.expander('Catatan teknis pelatihan'): st.write('\n\n'.join(messages))
    with st.expander('Hasil yang tersimpan pada notebook lama'):
        st.warning('Arsip, bukan hasil pipeline yang diperbaiki. Jangan membandingkan angka ini seolah-olah prosesnya sama. Kesimpulan notebook menyebut Linear dengan R² sekitar 0,779.')
        nbpath=ROOT/'Prediction_Car_Prize.ipynb'
        if nbpath.exists():
            nb=json.loads(nbpath.read_text(encoding='utf-8'))
            for cell in nb.get('cells',[]):
                src=''.join(cell.get('source',[]))
                if 'print(' in src and ('RMSE' in src or 'rmse' in src):
                    for out in cell.get('outputs',[]):
                        if out.get('output_type')=='stream': st.text(''.join(out.get('text',[])))
        else: st.caption('Notebook asli tidak tersedia di folder aplikasi.')

elif page == 'Prediksi Harga':
    st.write('Bandingkan hasil empat model untuk spesifikasi yang sama. Model yang dipakai tetap model hasil latihan 80%, sehingga sesuai dengan evaluasi.')
    st.warning('Simulasi akademik, bukan penawaran harga pasar. Satuan masukan mengikuti CSV; beberapa satuan belum terverifikasi. Kombinasi spesifikasi baru belum tentu realistis.')
    choice=st.selectbox('Gunakan mobil contoh sebagai nilai awal',df.index.tolist(),format_func=lambda i:f"{df.loc[i,'car_ID']} · {df.loc[i,'CarName']}")
    base=df.loc[choice]
    with st.form('prediksi'):
        values={}; cols=st.columns(3)
        for i,c in enumerate(NUM):
            values[c]=cols[i%3].number_input(f'{LABEL[c]} ({c})',min_value=float(df[c].min()),max_value=float(df[c].max()),value=float(base[c]),key=f'{choice}_{c}')
        for i,c in enumerate(CAT):
            opts=sorted(df[c].dropna().unique().tolist())
            values[c]=cols[i%3].selectbox(LABEL[c],opts,index=opts.index(base[c]),key=f'{choice}_{c}')
        submit=st.form_submit_button('Bandingkan prediksi 4 model',type='primary')
    if submit:
        row=pd.DataFrame([values]); results=[]
        for name,model in models.items():
            with warnings.catch_warnings():
                warnings.simplefilter('ignore'); value=float(model.predict(row)[0])
            results.append({'Model':name,'Prediksi harga':value})
        result=pd.DataFrame(results)
        for col,r in zip(st.columns(4),results): col.metric(r['Model'],f"{r['Prediksi harga']:,.2f}")
        chart(alt.Chart(result).mark_bar().encode(x='Model:N',y='Prediksi harga:Q',color=alt.Color('Model:N',legend=None),tooltip=['Model','Prediksi harga']).properties(height=290))
        st.caption(f'Model dengan RMSE uji terendah: {best.Model}. Selisih hasil antar-model bukan interval kepercayaan.')
        if any(r['Prediksi harga']<df.price.min() or r['Prediksi harga']>df.price.max() for r in results): st.warning('Ada hasil di luar rentang harga dataset. Nilainya ditampilkan apa adanya; jangan dianggap estimasi yang andal.')
        membership='latihan' if choice in train_idx else 'pengujian'
        st.caption(f'Mobil contoh berasal dari data {membership}. Menguji ulang contoh dataset bukan pengujian independen baru.')
        table(row); download(result,'hasil_simulasi.csv')

elif page == 'Kesimpulan':
    corr=df.select_dtypes('number').drop(columns=['car_ID','price']).corrwith(df.price).sort_values(key=abs,ascending=False)
    trainbest=scores[(scores.Model==best.Model)&(scores.Data=='Latih')].iloc[0]
    st.subheader('1 · Gambaran data')
    st.write(f'Dataset berisi {len(df)} mobil dan {len(df.columns)} kolom. Harga berkisar {df.price.min():,.2f}–{df.price.max():,.2f}, dengan median {df.price.median():,.2f}. Terdapat {int(df.isna().sum().sum())} sel kosong dan {int(df.duplicated().sum())} duplikat identik.')
    st.subheader('2 · Hubungan dengan harga')
    st.write('Tiga fitur dengan korelasi linear absolut terbesar: '+', '.join(f'{k} (r={v:.3f})' for k,v in corr.head(3).items())+'. Korelasi tidak membuktikan sebab-akibat atau kepentingan fitur dalam model.')
    st.subheader('3 · Hasil perbandingan empat model')
    st.success(f"{best.Model} terbaik pada pembagian pengujian ini berdasarkan RMSE {best.RMSE:,.2f}; R² {best['R²']:.3f}; MAE {best.MAE:,.2f}.")
    table(rank)
    st.write(f"R² latihan model tersebut {trainbest['R²']:.3f}, dibanding R² uji {best['R²']:.3f}; selisihnya {trainbest['R²']-best['R²']:.3f}. Penilaian model harus memperhatikan hasil uji, bukan hanya hasil latihan.")
    st.subheader('4 · Keterbatasan dan saran')
    st.markdown('''- Data relatif kecil dan hanya menggunakan satu pembagian latihan–pengujian; peringkat dapat berubah dengan pembagian lain.
- Data uji telah dipakai untuk membandingkan model. Penelitian lanjutan sebaiknya memakai cross-validation pada data latihan dan pengujian akhir independen.
- Hyperparameter belum dioptimalkan. Polynomial memiliki banyak fitur dan berisiko overfit.
- Tahun, sumber asli, dan mata uang belum terverifikasi; model tidak mewakili harga pasar saat ini.
- Kombinasi spesifikasi di luar pola latihan dapat menghasilkan estimasi buruk. Tambahkan data representatif, verifikasi metadata, dan evaluasi ulang sebelum pemakaian nyata.''')
    st.subheader('5 · Perbedaan dari laporan lama')
    st.write('Notebook menyatakan Linear terbaik dengan R² sekitar 0,779. Dashboard menghitung ulang tanpa Car_category yang berasal dari target, dengan persiapan hanya pada data latihan. Karena prosedurnya berubah, hasil baru tidak boleh diklaim sebagai reproduksi persis hasil lama.')
    with st.expander('Pegangan untuk presentasi'):
        st.markdown('''**Mengapa empat model?** Membandingkan model linear dasar, model lebih kompleks (Polynomial), dan dua model dengan regularisasi (Ridge dan Lasso).

**Apa arti R²?** Ukuran kecocokan dibandingkan menggunakan rata-rata target. Mendekati 1 lebih baik; bisa negatif; bukan persentase akurasi.

**Mengapa RMSE?** Mengukur kesalahan dalam satuan harga dan memberi bobot lebih besar pada kesalahan besar. Nilai terendah menjadi kriteria peringkat di dashboard.

**Mengapa hasil berubah?** Kebocoran informasi harga dihapus dan normalisasi dimasukkan ke pipeline agar evaluasi lebih dapat dipercaya.

**Apakah model terbaik pasti selalu benar?** Tidak. Kesimpulan terbatas pada dataset, pembagian data, dan pengaturan yang digunakan.''')

st.divider()
st.caption('CarLab · Sumber kerja: CarPrice.csv dan Prediction_Car_Prize.ipynb · Evaluasi dihitung ulang, bukan angka contoh.')
