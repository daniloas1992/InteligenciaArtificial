import random
from texttable import Texttable

file_amostras = open('Amostras.txt', 'r')
file_testes = open('Teste.txt', 'r')

QUANTIDADE_POR_AMOSTRA = 3
TAXA_APRENDIZAGEM = 0.01
QTD_MAX_EPOCAS = 2000
amostras = []
resultados = []
pesos = []
conjuntoTeste = []
resultados_testes = []

def ler(op):
    if op == 1:
        l = file_amostras.readline().split()
    else:
        l = file_testes.readline().split()

    if not l:
        return False

    for index, item in enumerate(l):
        l[index] = float(item)

    return l


def carregarDados():
    print('\n-> Carregando conjunto de amostras e resultados...')
    while True:

        line = ler(1)

        if not line:
            file_amostras.close()
            break

        amostras.append(line[0:QUANTIDADE_POR_AMOSTRA])
        resultados.append(line[QUANTIDADE_POR_AMOSTRA])

    print('\n-> Carregando conjunto de testes...')
    while True:

        line = ler(2)

        if not line:
            file_testes.close()
            break

        conjuntoTeste.append(line)

def gerarPesosIniciais():
    for i in range(QUANTIDADE_POR_AMOSTRA + 1):  # Inclui o peso do Bias
        pesos.append(round(random.random(), 2))


def funcaoDegrauBipolar(u):
    if (u < 0):
        return -1
    else:
        return 1

def separador(epoca):
    print('\n-----------------------------------------')
    print('\t\t\t\tÉpoca: %i' % (epoca))
    print('-----------------------------------------')


def treinar():
    print('\n-> Treinamento em andamento...')
    epoca = 0
    erro = True

    while (erro):

        erro = False
        #separador(epoca)

        for i in range(len(amostras)):
            '''
            print('Entradas: ', end = '')
            print((amostras[i]))
            print('Pesos: ', end = '')
            print(pesos[0:len(pesos)-1])
            print('Peso do Limiar: %f' % pesos[len(pesos)-1])
            print('Saída desejada: %d' % (resultados[i]))
            '''
            u = 0
            for j in range(len(amostras[i])):
                u += (pesos[j] * amostras[i][j])

            u = u - pesos[len(pesos) - 1]
            #print("\nU: %f" % (u))

            y = funcaoDegrauBipolar(u)
            #print("Y: %i" % (y))

            if resultados[i] != y:

                e = resultados[i] - y
                #print("E: %i\n" % (e))

                #print("Resultado diferente do esperado.\n")
                erro = True

                for k in range(len(amostras[i])):
                    pesos[k] = pesos[k] + TAXA_APRENDIZAGEM * e * amostras[i][k]


                pesos[len(pesos) - 1] = pesos[len(pesos) - 1] + TAXA_APRENDIZAGEM * e * (-1)# Ajusta o peso do limiar
                '''
                print('Pesos Atualizados: ', end = '')
                print(pesos[0:len(pesos)-1])
                print('Peso Limiar Atualizado: %f' % pesos[len(pesos)-1])
                '''
        epoca += 1

        if epoca > QTD_MAX_EPOCAS:
            print('\n-> Treinamento Atingiu Limite Máximo de Épocas!\n\n-> Total de épocas: %i' %epoca)
            break

        if not erro:
            print('\n-> Treinamento finalizado!\n\n-> Total de épocas: %i' %epoca)
            break



def descricaoOleo(resultado):

    if resultado == -1:
        return 'C1'
    else:
        return 'C2'

def testar():

    for i in range(len(conjuntoTeste)):

        u = 0
        for j in range(len(conjuntoTeste[i])):
            u += (pesos[j] * conjuntoTeste[i][j])
        u = u - pesos[len(pesos) - 1]

        resultados_testes.append(descricaoOleo(funcaoDegrauBipolar(u)))

def imprimirPesos(texto):
    print('\n-> %s:\n' %texto)
    print('\tW[0] = %f' % (pesos[len(pesos)-1]))
    for i in range(0,3,1):
        print('\tW[%i] = %f' % ((i+1), pesos[i]))

def imprimirResultados():
    print('\n\t\t\t\t\t RESULTADOS')
    table = Texttable()
    header = ['Amostra','Entradas','Classificação']

    for i in range(len(resultados_testes)):
        table.add_rows([header, [i+1, conjuntoTeste[i], resultados_testes[i]]])
    print(table.draw())

def main():
    carregarDados()
    gerarPesosIniciais()
    imprimirPesos('Pesos Iniciais')
    treinar()
    imprimirPesos('Pesos ajustados após treinamento')
    testar()
    imprimirResultados()

if __name__ == "__main__":
    main()