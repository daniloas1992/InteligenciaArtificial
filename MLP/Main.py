import math
import random
from texttable import Texttable
from datetime import datetime
import matplotlib.pyplot as plt

file_amostras = open('Amostras.txt', 'r')
file_testes = open('Teste.txt', 'r')

QTD_NEURONIOS_CAMADA_OCULTA = 15
QTD_NEURONIOS_CAMADA_SAIDA = 3
QUANTIDADE_POR_AMOSTRA = 5  # -> Somar +1 para incluir uma posição para o Bias
TAXA_APRENDIZAGEM = 0.1
QTD_MAX_EPOCAS = 5000
erroAdmissivel = 0.000001

amostras = []
resultados = []
pesos_camada_oculta = []
pesos_camada_saida = []
conjuntoTeste = []
conjuntoTesteResultados = []
resultados_testes = []
resultados_Y_testes = []
armazenaEQM = []

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


def carregar_dados():
    print('\n-> Carregando conjunto de amostras e resultados...')
    while True:

        line = ler(1)

        if not line:
            file_amostras.close()
            break

        amostras.append(line[0:QUANTIDADE_POR_AMOSTRA])
        resultados.append(line[QUANTIDADE_POR_AMOSTRA:])

    print('\n-> Carregando conjunto de testes...')
    while True:

        line = ler(2)

        if not line:
            file_testes.close()
            break

        conjuntoTeste.append(line[0:QUANTIDADE_POR_AMOSTRA])
        conjuntoTesteResultados.append(line[QUANTIDADE_POR_AMOSTRA:])


def gerar_pesos_iniciais (pesos, qtdNeuronios, qtdEntradas):

    for i in range(qtdNeuronios):
        pesos_gerados = []
        for j in range(qtdEntradas):
            pesos_gerados.append(round(random.random(), 2))
        pesos.append(pesos_gerados)

def funcao_logistica(u):
    return (1 / (1 + (pow(math.e, -u))))


def calcular_saida_y(u):
    saidas = []
    for i in range(len(u)):
        saidas.append(funcao_logistica(u[i]))

    return saidas


def imprimirPesos(texto, pesos, tipo):
    print('\n-> %s:' % texto)

    if tipo == 'ENTRADA':
        for i in range(QTD_NEURONIOS_CAMADA_OCULTA):
            print('\n%s: %i\n' % ('Camada Oculta - Neurônio', i + 1))

            for j in range(QUANTIDADE_POR_AMOSTRA):
                print('\tW[%i][%i] = %f' % (i + 1, j, pesos[i][j]))

    elif tipo == 'SAIDA':
        for i in range(QTD_NEURONIOS_CAMADA_SAIDA):
            print('\n%s: %i\n' % ('Camada Saída - Neurônio', QTD_NEURONIOS_CAMADA_OCULTA + i + 1))
            for j in range(QTD_NEURONIOS_CAMADA_OCULTA + 1):
                print('\tW[%i][%i] = %f' % (i + QTD_NEURONIOS_CAMADA_OCULTA + 1, j, pesos[i][j]))


def calcular_ativacao_u(pesos, qtdNeuronios, entradas):

    somatorio = []

    for i in range(qtdNeuronios):
        u = 0
        for j in range(len(entradas)):
            #print('%f * %f '%(pesos[i][j], entradas[j]))
            u += pesos[i][j] * entradas[j]
        somatorio.append(u)

    return somatorio


def calcular_erro_camada_saida(qtdNeuronios, resultados_obtidos, resultados_esperados):

    erros_saida = []
    for i in range(qtdNeuronios):
        #print('%f - %f '%(resultados_esperados[i], resultados_obtidos[i]))
        erros_saida.append(-(resultados_esperados[i]-resultados_obtidos[i]))

    return erros_saida

def calcular_sigma_saida(qtdNeuronios, saidasObtidas, errosSaidas):

    erros = []
    for i in range(qtdNeuronios):
        desvio = (saidasObtidas[i] * (1 - saidasObtidas[i])) * errosSaidas[i]
        erros.append(desvio)

    return erros

def calcular_erros_saida_relacao_entradas(qtdNeuronios, qtdEntradas, desvio_camada_saida, y_camada_oculta):

    erros = []
    for i in range(qtdNeuronios):
        e = []
        for j in range(qtdEntradas):
           #print('%f * %f' %(desvio_camada_saida[i], y_camada_oculta[j]))
           e.append(desvio_camada_saida[i] * y_camada_oculta[j])

        erros.append(e)

    return erros


def calcular_erros_camada_oculta(qtdNeuronios, qtdEntradas, erro_saida):

    erros = []

    for i in range(qtdNeuronios):
        e = 0.0
        for j in range(qtdEntradas):
            #print('%f * %f'%(erro_saida[j],pesos_camada_saida[j][i+1]))
            e+=(erro_saida[j] * pesos_camada_saida[j][i+1])
            #erros.append(e)
        erros.append(e)

    return erros


def calcular_erro_saida_oculta(qtdNeuronios, y_camada_oculta):
    erros = []
    for saida in y_camada_oculta[1: len(y_camada_oculta): 1]:  # Na fase de backpropagation o bias de entrada na camada de saída não interfere então deve ser pulado
        erros.append(saida * (1 - saida))

    return erros


def calcular_erros_oculto_relacao_entradas(qtdNeuronios, qtdENtradas, erros_ocultos_em_relacao_entradas, erro_peso_neuronio, amostras):

    erros = []
    for i in range(qtdNeuronios):
        e = []
        for j in range(qtdENtradas):
            e.append(erros_ocultos_em_relacao_entradas[i] * erro_peso_neuronio[i] * amostras[j])

        erros.append(e)

    return erros


def atualizar_pesos(qtdNeuronios, qtdEntradas, erros, peso):

    for i in range(qtdNeuronios):
        for j in range(qtdEntradas):
            peso[i][j] = peso[i][j] - TAXA_APRENDIZAGEM * erros[i][j]


def calcular_erro_amostra(resultado_obtido, resultado_esperado):
    return ((1/2) * (math.pow((resultado_esperado - resultado_obtido),2)))


def calcularEQM():

    saidas_obtidas = []

    for posicao in range(len(amostras)):

        u_camada_oculta = calcular_ativacao_u(pesos_camada_oculta, QTD_NEURONIOS_CAMADA_OCULTA, amostras[posicao])
        y_camada_oculta = calcular_saida_y(u_camada_oculta)

        y_camada_oculta.insert(0, -1.0)  # Adiciona o multiplicador do bias na primeira posição

        u_camada_saida = calcular_ativacao_u(pesos_camada_saida, QTD_NEURONIOS_CAMADA_SAIDA, y_camada_oculta)
        y_camada_saida = calcular_saida_y(u_camada_saida)

        saidas_obtidas.append(y_camada_saida)

    somatorioErros = [0.0] * QTD_NEURONIOS_CAMADA_SAIDA

    for i in range(len(saidas_obtidas)):
        for j in range(QTD_NEURONIOS_CAMADA_SAIDA):
            #print('%f %f' %(saidas_obtidas[i][j], resultados[i][j]))
            somatorioErros[j] += calcular_erro_amostra(saidas_obtidas[i][j], resultados[i][j])

    return (sum(somatorioErros) / len(amostras))

def treinar():

    print('\n-> Treinamento em andamento...')
    total_epocas = 0
    erroAtual = 0
    erroAnterior = 1000

    while (abs(erroAtual - erroAnterior) > erroAdmissivel):  # Enquanto tiver erro fica treinando

        erroAnterior = erroAtual

        for posicao in range(len(amostras)):

            ##############################  ETAPA FORWARD  ##############################

            u_camada_oculta = calcular_ativacao_u(pesos_camada_oculta, QTD_NEURONIOS_CAMADA_OCULTA, amostras[posicao])
            y_camada_oculta = calcular_saida_y(u_camada_oculta)

            y_camada_oculta.insert(0, -1.0) # Adiciona o multiplicador do bias na primeira posição

            u_camada_saida = calcular_ativacao_u(pesos_camada_saida, QTD_NEURONIOS_CAMADA_SAIDA, y_camada_oculta)
            y_camada_saida = calcular_saida_y(u_camada_saida)

            ##############################  ETAPA BACKWARD  ##############################

            # Calculo dos erros da camada de saída
            erros_camada_saida = calcular_erro_camada_saida(QTD_NEURONIOS_CAMADA_SAIDA, y_camada_saida, resultados[posicao])
            desvio_camada_saida = calcular_sigma_saida(QTD_NEURONIOS_CAMADA_SAIDA, y_camada_saida, erros_camada_saida)
            erros_saida_relacao_entradas = calcular_erros_saida_relacao_entradas(QTD_NEURONIOS_CAMADA_SAIDA, (QTD_NEURONIOS_CAMADA_OCULTA+1), desvio_camada_saida, y_camada_oculta)

            # Calculo dos erros da camada oculta
            erros_camada_oculta = calcular_erros_camada_oculta(QTD_NEURONIOS_CAMADA_OCULTA, QTD_NEURONIOS_CAMADA_SAIDA, desvio_camada_saida)
            erro_saida_oculta = calcular_erro_saida_oculta(QTD_NEURONIOS_CAMADA_OCULTA, y_camada_oculta)
            erros_oculto_relacao_entradas = calcular_erros_oculto_relacao_entradas(QTD_NEURONIOS_CAMADA_OCULTA, QUANTIDADE_POR_AMOSTRA, erros_camada_oculta, erro_saida_oculta, amostras[posicao])


            # Atualização dos pesos
            atualizar_pesos(QTD_NEURONIOS_CAMADA_SAIDA, (QTD_NEURONIOS_CAMADA_OCULTA+1), erros_saida_relacao_entradas, pesos_camada_saida) # Camada de Saída
            atualizar_pesos(QTD_NEURONIOS_CAMADA_OCULTA, QUANTIDADE_POR_AMOSTRA, erros_oculto_relacao_entradas, pesos_camada_oculta) # Camada Oculta

        total_epocas += 1

        if total_epocas > QTD_MAX_EPOCAS:
            print('\n-> Treinamento Atingiu Limite Máximo de Épocas!\n\n-> Total de épocas: %i' % total_epocas)
            break

        erroAtual = calcularEQM()
        armazenaEQM.append(erroAtual)

    return total_epocas

def identificar_conservante(resultados):

    padrao = []
    for i in range(len(resultados)):
        if(resultados[i]) >= 0.5:
            padrao.append(1)
        else:
            padrao.append(0)

    if padrao[0] == 1:
        return 'A'

    if padrao[1] == 1:
        return 'B'

    if padrao[2] == 1:
        return 'C'

    return 'Desconhecido'

def testar():

    for posicao in range(len(conjuntoTeste)):
        u_camada_oculta = calcular_ativacao_u(pesos_camada_oculta, QTD_NEURONIOS_CAMADA_OCULTA, conjuntoTeste[posicao])
        y_camada_oculta = calcular_saida_y(u_camada_oculta)

        y_camada_oculta.insert(0, -1.0)  # Adiciona o multiplicador do bias na primeira posição

        u_camada_saida = calcular_ativacao_u(pesos_camada_saida, QTD_NEURONIOS_CAMADA_SAIDA, y_camada_oculta)
        y_camada_saida = calcular_saida_y(u_camada_saida)

        resultados_Y_testes.append(y_camada_saida)
        resultados_testes.append(identificar_conservante(y_camada_saida))

def imprimirResultados2():
    print('\n\t\t\t\t\t RESULTADOS')
    table = Texttable()
    header = ['Amostra','X1','X2','X3','X4','d1','d2','d3','Y1pós','Y2pós','Y3pós', 'Conservante']

    for i in range(len(resultados_testes)):
        table.add_rows([header, [i + 1, conjuntoTeste[i][1],conjuntoTeste[i][2],conjuntoTeste[i][3],conjuntoTeste[i][4],conjuntoTesteResultados[i][0],conjuntoTesteResultados[i][1],conjuntoTesteResultados[i][2], resultados_Y_testes[i][0],resultados_Y_testes[i][1],resultados_Y_testes[i][2], resultados_testes[i]]])
    print(table.draw())

def imprimirResultados():
    print('\n\t\t\t\t\t\t\t RESULTADOS')
    table = Texttable()
    header = ['Amostra', 'Entradas', 'Saidas Esperdas','Saidas Obtidas', 'Conservante']

    for i in range(len(resultados_testes)):
        table.add_rows([header, [i + 1, conjuntoTeste[i][1:], conjuntoTesteResultados[i], resultados_Y_testes[i], resultados_testes[i]]])
    print(table.draw())

def gerarGrafico():
    totalEpocas = []
    for i in range(len(armazenaEQM)):
        totalEpocas.append(i)

    plt.plot(totalEpocas, armazenaEQM)
    plt.title('Grafico Eqm X Épocas')
    plt.xlabel('Épocas')
    plt.ylabel('Eqm')
    plt.show()

def exibirTotalEpocas(total_epocas):
    print('\nTotal de Épocas: %i'% total_epocas)

def imprimirTempoProcessamento(start_time,end_time):
    print('\nTempo de treinamento: {}\n'.format(end_time - start_time))
def main():

    carregar_dados()
    gerar_pesos_iniciais(pesos_camada_oculta, QTD_NEURONIOS_CAMADA_OCULTA, QUANTIDADE_POR_AMOSTRA)
    gerar_pesos_iniciais(pesos_camada_saida, QTD_NEURONIOS_CAMADA_SAIDA, QTD_NEURONIOS_CAMADA_OCULTA + 1)
    imprimirPesos('Pesos Iniciais Camada Entrada', pesos_camada_oculta, 'ENTRADA')
    imprimirPesos('Pesos Iniciais Camada Saída', pesos_camada_saida, 'SAIDA')
    start_time = datetime.now()
    epocas = treinar()
    end_time = datetime.now()
    imprimirPesos('Pesos Finais Camada Entrada', pesos_camada_oculta, 'ENTRADA')
    imprimirPesos('Pesos Finais Camada Saída', pesos_camada_saida, 'SAIDA')
    imprimirTempoProcessamento(start_time, end_time)
    exibirTotalEpocas(epocas)
    testar()
    imprimirResultados()
    gerarGrafico()

if __name__ == "__main__":
    main()
