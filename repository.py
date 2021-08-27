'.\tf_2\Scripts\activate'
"C:\Users\Max Power\OneDrive\ponte\programmi\python\virtual_env\tf_2\Scripts\activate"

import sys
sys.path.insert(0, 'C:/Users/Max Power/OneDrive/ponte/programmi/python/progetto2/AJ_lib')
from AJ_draw import disegna as ds

pip install xxx --upgrade
pip install "xxx==1.2"

pip show numpy #per vedere la versione del pacchetto (vale per tutti i pacchetti)

data_row = pd.read_csv(filename, delimiter=';', encoding="ISO-8859-1")
data_final_missing.to_csv('final_disposition_missing.csv', sep=';')
def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

col_num = [i for i in range(numb_of_spect)]
x = np.arange(len(prob_have_class.columns))
x1 = np.linspace(0, 10, 8, endpoint=True)

G = np.zeros_like(reward)#Return an array of zeros with the same shape and type as a given array.
arr = numpy.asarray(lst) #converte una lista in un numpy puntando alla lista, senza creare un duplicato
arr = numpy.array(lst) #converte una lista in un numpy creando un duplicato della lista

import itertools
power_vet = list(map(round, power_vet, itertools.repeat(2, len(power_vet))))

data = data.astype({'raggio_nm': 'int'})
data_compare['col'] = data_compare['col'].astype(float)
file_PL = file_PL.apply(pd.to_numeric, downcast='float')
prices[col] = prices[col].cat.codes.astype("int16")# converte una colonna con tipo category in int

x_train = x_train.reset_index(drop = True)
data.set_index('date', inplace = True)

data.rename(columns={'pop':'population'}, inplace=True)

#riordina le colonne mettendo quelle selezionate davanti
data = data[['anno', 'company', 'sub']+[c for c in data if c not in ['anno', 'company', 'sub']]]

#uso delle maschere
mask = df_over_soglia_update['evento predetto'] > 0
df_over_soglia_update.loc[mask,'evento predetto'] = 1

data_row['upload year'] = data_row['DOI'].str.split(".", expand = True)[3]

cm = [plt.cm.rainbow(i) for i in np.linspace(0, 1.0, df_compare.T.shape[1] + 1)]

plt.xticks(rotation=90)


END_EPSILON_DECAYING = EPISODES // 2 #rida sempre un intero
episode % SHOW_EVERY == 0 #da il residuo di una divisione


compila: shift + ctrl + B
outline: alt + o
debugger: creare un breakpoint
commenta: ctrl + /

linea di comando di atom: crtl + shift + p

controlla le differenze tra 2 testi aperti uno affianco all'altro: ctrl + alt + t
controllo dei brunch: crtl + shift + p -> show

hydrogen run selected: shift + enter
hydrogen run all: ctrl + shift + alt + enter
hydrogen run cell: alt + shift + enter
hydrogen clear: ctrl + shift + backspace

%matplotlib inline # decoratore per usare plot con hydrogen

# C:\Users\Max Power\AppData\Local\Programs\Python\Python36\python.exe
# C:\Users\Max Power\anaconda3\python.exe
# C:\Users\Max Power\AppData\Local\Microsoft\WindowsApps\python.exe

#fix the multicursor click
Open the console (Ctrl + Shift + I) (Alt + Cmd + I on MacOS)
run atom.config.set('core.editor.multiCursorOnClick', true);
