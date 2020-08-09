# Selecionar somente valores inteiros e floats
df_test = df_test.select_dtypes(include=['int64','float64'])

# Salvar os dados das inscrições
df_resposta['NU_INSCRICAO'] = df_test['NU_INSCRICAO']
df_test.drop('NU_INSCRICAO', axis=1, inplace=True)

# Preencher valores nulos com o valor médio - Tratamento das notas de provas corrompidas
df_train['NU_NOTA_CN'].fillna(df_train['NU_NOTA_CN'].mean(), inplace=True)
df_train['NU_NOTA_CH'].fillna(df_train['NU_NOTA_CH'].mean(), inplace=True)
df_train['NU_NOTA_REDACAO'].fillna(df_train['NU_NOTA_REDACAO'].mean(), inplace=True)
df_train['NU_NOTA_LC'].fillna(df_train['NU_NOTA_LC'].mean(), inplace=True)
df_train['NU_NOTA_MT'].fillna(df_train['NU_NOTA_MT'].mean(), inplace=True)
df_test['NU_NOTA_CN'].fillna(df_train['NU_NOTA_CN'].mean(), inplace=True)
df_test['NU_NOTA_CH'].fillna(df_train['NU_NOTA_CH'].mean(), inplace=True)
df_test['NU_NOTA_REDACAO'].fillna(df_train['NU_NOTA_REDACAO'].mean(), inplace=True)
df_test['NU_NOTA_LC'].fillna(df_train['NU_NOTA_LC'].mean(), inplace=True)

features = df.columns.difference(['Class'])

X = df[features].values
y = df['Class'].values

df = result.DataFrame()
df = df.apply(pd.to_numeric)
df.head()

corrs = df.corr()['class_var'].abs()
columns = corrs[corrs > .01].index
corrs = corrs.filter(columns)
corrs
