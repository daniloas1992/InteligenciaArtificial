import math
import random
from texttable import Texttable
import matplotlib.pyplot as plt

file_amostras = open('Amostras.txt', 'r')
file_testes = open('Teste.txt', 'r')

QTD_NEURONIOS_CAMADA_OCULTA = 10
QTD_NEURONIOS_CAMADA_SAIDA = 1
QUANTIDADE_POR_AMOSTRA = 4  # -> Já inclui uma posição para o Bias
TAXA_APRENDIZAGEM = 0.1
QTD_MAX_EPOCAS = 5000
erroAdmissivel = 0.000001

amostras = []
resultados = []
pesos_camada_entrada_oculta = []
pesos_camada_oculta_saida = []
conjuntoTeste = []
conjuntoTesteResultados = []
resultados_testes = []
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

        conjuntoTeste.append(line[0:QUANTIDADE_POR_AMOSTRA])
        conjuntoTesteResultados.append(line[QUANTIDADE_POR_AMOSTRA])


def gerarPesosCamadaOculta():
    for coluna in range(QTD_NEURONIOS_CAMADA_OCULTA):
        pesos_gerados = []
        for linha in range(QUANTIDADE_POR_AMOSTRA):
            pesos_gerados.append(round(random.random(), 2))
        pesos_camada_entrada_oculta.append(pesos_gerados)


def gerarPesosCamadaSaida():
    for coluna in range(QTD_NEURONIOS_CAMADA_SAIDA):
        pesos_gerados = []
        for linha in range(QTD_NEURONIOS_CAMADA_OCULTA + 1):
            pesos_gerados.append(round(random.random(), 2))
        pesos_camada_oculta_saida.append(pesos_gerados)


def funcaoLogisitica(u):
    return (1 / (1 + (pow(math.e, -u))))


def calcular_somatorio_camada_oculta(posicao):

    somatorio_ocultas = []
    for i in range(QTD_NEURONIOS_CAMADA_OCULTA):
        uH = 0.0
        for j in range(QUANTIDADE_POR_AMOSTRA):
            uH += pesos_camada_entrada_oculta[i][j] * amostras[posicao][j]
        somatorio_ocultas.append(uH)

    return somatorio_ocultas


def funcao_logistica(u):
    saidas = []
    for i in range(len(u)):
        saidas.append(funcaoLogisitica(u[i]))

    return saidas


def calcular_somatorio_camada_saida(y_camada_oculta):

    somatorios_saida = []
    for i in range(QTD_NEURONIOS_CAMADA_SAIDA):
        uO = 0.0

        for j in range(len(y_camada_oculta)):
            uO += pesos_camada_oculta_saida[i][j+1] * y_camada_oculta[j]

        uO += (pesos_camada_oculta_saida[i][0] * (-1))  # Bias da camada de saida

        somatorios_saida.append(uO)

    return somatorios_saida


def calcular_erros_camada_saida(y_camada_saida, resultados_esperados):

    erros_saida = []
    for i in range(QTD_NEURONIOS_CAMADA_SAIDA):
        erros_saida.append(-(resultados_esperados-y_camada_saida[i]))

    return erros_saida

def calcular_erro_derivada_logistica(y_camada_saida):

    erros = []
    for i in range(QTD_NEURONIOS_CAMADA_SAIDA):
        erros.append(y_camada_saida[i] * (1 - y_camada_saida[i]))

    return erros

def calcular_erros_entradas(erros_total, erros_entradas, y_camada_oculta):

    erros = []
    for i in range(QTD_NEURONIOS_CAMADA_SAIDA):
        erros.append(erros_total[i] * erros_entradas[i] * (-1.0))  # Erro do Bias
        for j in range(QTD_NEURONIOS_CAMADA_OCULTA):
            erros.append(erros_total[i] * erros_entradas[i] * y_camada_oculta[j])

    return erros

def atualizar_pesos_camada_saida(erros_entradas_em_relacao_saida):

    for i in range(QTD_NEURONIOS_CAMADA_SAIDA):
        for j in range(QTD_NEURONIOS_CAMADA_OCULTA):
            pesos_camada_oculta_saida[i][j] = pesos_camada_oculta_saida[i][j] - TAXA_APRENDIZAGEM * erros_entradas_em_relacao_saida[j]

def calcular_erro_oculto_saida(erros_total_em_relacao_saida, erros_saida_em_relacao_total_entradas):

    erros = []
    for i in range(QTD_NEURONIOS_CAMADA_SAIDA):
        erros.append(erros_total_em_relacao_saida[i] * erros_saida_em_relacao_total_entradas[i])

    return erros

def calcular_erros_oculto_entradas(erros_ocultos_em_relacao_saidas):

    erros = []

    for i in range(QTD_NEURONIOS_CAMADA_SAIDA):
        #e = []
        for j in range(QTD_NEURONIOS_CAMADA_OCULTA):
            #e.append(erros_ocultos_em_relacao_saidas[i]* pesos_camada_oculta_saida[i][j+1])
            erros.append(erros_ocultos_em_relacao_saidas[i]* pesos_camada_oculta_saida[i][j+1])
        #erros.append(e)

    return erros

def calcular_erros_derivada_logistica_camada_oculta(y_camada_oculta):

    erros = []
    for i in range(QTD_NEURONIOS_CAMADA_OCULTA):
        erros.append(y_camada_oculta[i] * (1 - y_camada_oculta[i]))

    return erros

def calcular_erros_entradas_oculto(erros_ocultos_em_relacao_entradas, erro_peso_neuronio, posicao):

    erros = []
    for i in range(QTD_NEURONIOS_CAMADA_OCULTA):
        e = []
        for j in range(QUANTIDADE_POR_AMOSTRA):
            e.append(erros_ocultos_em_relacao_entradas[i] * erro_peso_neuronio[i] * amostras[posicao][j])

        erros.append(e)

    return erros

def atualizar_pesos_camada_oculta(erros_entradas_em_relacao_ocultos):

    for i in range(QTD_NEURONIOS_CAMADA_OCULTA):
        for j in range(QUANTIDADE_POR_AMOSTRA):
            pesos_camada_entrada_oculta[i][j] = pesos_camada_entrada_oculta[i][j] - TAXA_APRENDIZAGEM * erros_entradas_em_relacao_ocultos[i][j]


def calcular_erro_amostra(resultado_obtido, resultado_esperado):
    return ((1/2) * (math.pow((resultado_esperado - resultado_obtido),2)))


def calcularEQM():

    saidas_obtidas = []

    for posicao in range(len(amostras)):
        u_camada_oculta = calcular_somatorio_camada_oculta(posicao)
        y_camada_oculta = funcao_logistica(u_camada_oculta)
        u_camada_saida = calcular_somatorio_camada_saida(y_camada_oculta)
        y_camada_saida = funcao_logistica(u_camada_saida)
        saidas_obtidas.append(y_camada_saida[0])

    somatorioErros = 0
    for i in range(len(saidas_obtidas)):
        somatorioErros += calcular_erro_amostra(saidas_obtidas[i], resultados[i])

    return (somatorioErros / len(amostras))


def treinar():
    print('\n-> Treinamento em andamento...')
    epoca = 0
    erroAtual = 0
    erroAnterior = 100

    while (abs(erroAtual - erroAnterior) > erroAdmissivel):  # Enquanto tiver erro fica treinando

        erroAnterior = erroAtual

        for posicao in range(len(amostras)):

            #############  ETAPA FORWARD  #############

            u_camada_oculta = calcular_somatorio_camada_oculta(posicao)
            y_camada_oculta = funcao_logistica(u_camada_oculta)
            u_camada_saida = calcular_somatorio_camada_saida(y_camada_oculta)
            y_camada_saida = funcao_logistica(u_camada_saida)

            #############  ETAPA BACKWARD  #############

            # Calculo dos erros da camada de saída

            erros_total_em_relacao_saida = calcular_erros_camada_saida(y_camada_saida, resultados[posicao])  # Erro dos neurônios de saída em relação ao total da rede
            erros_saida_em_relacao_total_entradas = calcular_erro_derivada_logistica(y_camada_saida)
            erros_entradas_em_relacao_saida = calcular_erros_entradas(erros_total_em_relacao_saida, erros_saida_em_relacao_total_entradas, y_camada_oculta)

            # Calculo dos erros da camada oculta

            erros_ocultos_em_relacao_saidas = calcular_erro_oculto_saida(erros_total_em_relacao_saida, erros_saida_em_relacao_total_entradas)
            erros_ocultos_em_relacao_entradas = calcular_erros_oculto_entradas(erros_ocultos_em_relacao_saidas)
            erro_peso_neuronio = calcular_erros_derivada_logistica_camada_oculta(y_camada_oculta)
            erros_entradas_em_relacao_ocultos = calcular_erros_entradas_oculto(erros_ocultos_em_relacao_entradas, erro_peso_neuronio, posicao)

            # Atualização dos pesos

            atualizar_pesos_camada_saida(erros_entradas_em_relacao_saida)
            atualizar_pesos_camada_oculta(erros_entradas_em_relacao_ocultos)

        epoca +=1

        if epoca > QTD_MAX_EPOCAS:
            print('\n-> Treinamento Atingiu Limite Máximo de Épocas!\n\n-> Total de épocas: %i' % epoca)
            break

        erroAtual = calcularEQM()
        armazenaEQM.append(erroAtual)
        #print('Época: %i \t Anterior: %f \t Atual: %f \t Erro: %f' % (epoca, erroAnterior, erroAtual, abs(erroAtual - erroAnterior)))

    return epoca


def calcular_somatorio_camada_oculta_teste(posicao):

    somatorio_ocultas = []
    for i in range(QTD_NEURONIOS_CAMADA_OCULTA):
        uH = 0.0
        for j in range(QUANTIDADE_POR_AMOSTRA):
            uH += pesos_camada_entrada_oculta[i][j] * conjuntoTeste[posicao][j]
        somatorio_ocultas.append(uH)

    return somatorio_ocultas

def testar():

    for posicao in range(len(conjuntoTeste)):

        u_camada_oculta = calcular_somatorio_camada_oculta_teste(posicao)
        y_camada_oculta = funcao_logistica(u_camada_oculta)
        u_camada_saida = calcular_somatorio_camada_saida(y_camada_oculta)
        y_camada_saida = funcao_logistica(u_camada_saida)

        resultados_testes.append(y_camada_saida[0])


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


def imprimirResultados():
    print('\n\t\t\t\t\t RESULTADOS')
    table = Texttable()
    header = ['Amostra', 'Entradas', 'Energia Absorvida']

    for i in range(len(resultados_testes)):
        table.add_rows([header, [i + 1, conjuntoTeste[i], resultados_testes[i]]])
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

def calcular_variancia():

     media = sum(resultados_testes) / len(resultados_testes)

     soma = 0
     for i in range(len(resultados_testes)):
         soma += math.pow((resultados_testes[i] - media),2)

     variancia = (soma / len(resultados_testes))

     print('Variância: %f'% variancia)


def calcular_erro_relativo():

    erros_absolutos = []

    for i in range(len(conjuntoTesteResultados)):
        erros_absolutos.append(abs(conjuntoTesteResultados[i] - resultados_testes[i]))

    erro_relativo = sum(erros_absolutos) / len(conjuntoTesteResultados)
    print('Erro Relativo: %f '% erro_relativo)


def exibirEQM():
    print('Erro Quadratico Médio: %f'% armazenaEQM[len(armazenaEQM)-1])

def exibirTotalEpocas(total_epocas):
    print('\nTotal de Épocas: %i'% total_epocas)

def main():
    carregarDados()
    gerarPesosCamadaOculta()
    imprimirPesos('Pesos Iniciais Camada Entrada', pesos_camada_entrada_oculta, 'ENTRADA')
    gerarPesosCamadaSaida()
    imprimirPesos('Pesos Iniciais Camada Saída', pesos_camada_oculta_saida, 'SAIDA')
    total_epocas = treinar()
    imprimirPesos('Pesos Finais Camada Entrada', pesos_camada_entrada_oculta, 'ENTRADA')
    imprimirPesos('Pesos Finais Camada Saída', pesos_camada_oculta_saida, 'SAIDA')
    exibirTotalEpocas(total_epocas)
    testar()
    imprimirResultados()
    gerarGrafico()
    exibirEQM()
    calcular_variancia()
    calcular_erro_relativo()


if __name__ == "__main__":
    main()
