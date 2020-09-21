import math
import random
from texttable import Texttable
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter)

file_amostras = open('Amostras.txt', 'r')
file_testes = open('Teste.txt', 'r')

QTD_NEURONIOS_CAMADA_OCULTA = 25
QTD_NEURONIOS_CAMADA_SAIDA = 1
TIME_DELAY = 16  # -> Somar +1 para incluir uma posição para o Bias
TAXA_APRENDIZAGEM = 0.1
QTD_MAX_EPOCAS = 5000
ERRO_ADMISSIVEL = 0.000005
FATOR_MOMENTUM = 0.8

amostras = []
resultados = []
pesos_camada_oculta = []
pesos_camada_oculta_anterior = []
pesos_camada_saida = []
pesos_camada_saida_anterior = []
conjuntoTeste = []
conjuntoTesteResultados = []
resultados_testes = []
resultados_Y_testes = []
armazenaEQM = []


def ler_arquivo(op):

    valores = []

    if op == 1:

        while True:
            valor = file_amostras.readline().split()

            if not valor:
                break

            valores.append(valor[0])

        file_amostras.close()

    else:
        while True:
            valor = file_testes.readline().split()

            if not valor:
                break

            valores.append(valor[0])

        file_testes.close()

    for index, item in enumerate(valores):
        valores[index] = float(item)

    return valores


def montar_amostras(valores,entradas):

    for i in range(len(valores)-TIME_DELAY+1):
        v = valores[i:i+TIME_DELAY-1]
        v.reverse()
        v.insert(0, -1.0)  # Adiciona o multiplicador do bias na primeira posição
        entradas.append(v)


def montar_resultados(valores, saidas):

    for i in valores[TIME_DELAY-1::1]:
        v = []
        v.append(i)
        saidas.append(v)

def carregar_dados():

    print('\n-> Carregando conjunto de amostras...')
    auxiliar_valores = ler_arquivo(1)
    montar_amostras(auxiliar_valores,amostras)
    montar_resultados(auxiliar_valores, resultados)

    print('\n-> Carregando conjunto de testes...')
    saida_final_amostras = auxiliar_valores[len(auxiliar_valores) - TIME_DELAY + 1:]
    saida_final_amostras.reverse()
    auxiliar_valores = ler_arquivo(2)

    for i in range(len(saida_final_amostras)): #Insere o último conjunto de amostra no começo do conjunto de teste
        auxiliar_valores.insert(0,saida_final_amostras[i])

    montar_amostras(auxiliar_valores,conjuntoTeste)
    montar_resultados(auxiliar_valores, conjuntoTesteResultados)

def gerar_pesos_iniciais (pesos, qtdNeuronios, qtdEntradas, zerar):

    for i in range(qtdNeuronios):
        pesos_gerados = []
        for j in range(qtdEntradas):
            if(zerar):
                pesos_gerados.append(0.0)
            else:
                pesos_gerados.append(round(random.random(), 2))
        pesos.append(pesos_gerados)


def funcao_logistica(u):
    return (1 / (1 + (pow(math.e, -u))))


def calcular_saida_y(u):
    saidas = []
    for i in range(len(u)):
        saidas.append(funcao_logistica(u[i]))

    return saidas


def imprimir_pesos(texto, pesos, tipo):
    print('\n-> %s:' % texto)

    if tipo == 'ENTRADA':
        for i in range(QTD_NEURONIOS_CAMADA_OCULTA):
            print('\n%s: %i\n' % ('Camada Oculta - Neurônio', i + 1))

            for j in range(TIME_DELAY):
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


def atualizar_pesos(qtdNeuronios, qtdEntradas, erros, peso, peso_anterior):

    for i in range(qtdNeuronios):
        for j in range(qtdEntradas):
            peso_atual = peso[i][j]
            peso[i][j] = peso[i][j] - TAXA_APRENDIZAGEM * erros[i][j] - (FATOR_MOMENTUM * (peso[i][j] - peso_anterior[i][j]))
            peso_anterior[i][j] = peso_atual


def calcular_erro_amostra(resultado_obtido, resultado_esperado):
    return ((1/2) * (math.pow((resultado_esperado - resultado_obtido),2)))


def calcular_eqm():

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

    while (abs(erroAtual - erroAnterior) > ERRO_ADMISSIVEL):  # Enquanto tiver erro fica treinando

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
            erros_oculto_relacao_entradas = calcular_erros_oculto_relacao_entradas(QTD_NEURONIOS_CAMADA_OCULTA, TIME_DELAY, erros_camada_oculta, erro_saida_oculta, amostras[posicao])


            # Atualização dos pesos
            atualizar_pesos(QTD_NEURONIOS_CAMADA_SAIDA, (QTD_NEURONIOS_CAMADA_OCULTA+1), erros_saida_relacao_entradas, pesos_camada_saida, pesos_camada_saida_anterior) # Camada de Saída
            atualizar_pesos(QTD_NEURONIOS_CAMADA_OCULTA, TIME_DELAY, erros_oculto_relacao_entradas, pesos_camada_oculta, pesos_camada_oculta_anterior) # Camada Oculta

        total_epocas += 1

        if total_epocas > QTD_MAX_EPOCAS:
            print('\n-> Treinamento Atingiu Limite Máximo de Épocas!\n\n-> Total de épocas: %i' % total_epocas)
            break

        erroAtual = calcular_eqm()
        armazenaEQM.append(erroAtual)

    return total_epocas


def testar():

    for posicao in range(len(conjuntoTeste)):
        u_camada_oculta = calcular_ativacao_u(pesos_camada_oculta, QTD_NEURONIOS_CAMADA_OCULTA, conjuntoTeste[posicao])
        y_camada_oculta = calcular_saida_y(u_camada_oculta)

        y_camada_oculta.insert(0, -1.0)  # Adiciona o multiplicador do bias na primeira posição

        u_camada_saida = calcular_ativacao_u(pesos_camada_saida, QTD_NEURONIOS_CAMADA_SAIDA, y_camada_oculta)
        y_camada_saida = calcular_saida_y(u_camada_saida)

        resultados_testes.append(y_camada_saida)


def imprimir_resultados():
    print('\n\t\t\t\t\t\t\t RESULTADOS')
    table = Texttable()
    header = ['Amostra', 'Entradas', 'Saidas Esperdas','Saidas Obtidas']

    for i in range(len(resultados_testes)):
        table.add_rows([header, [i + 1, conjuntoTeste[i][1:], conjuntoTesteResultados[i], resultados_testes[i]]])
    print(table.draw())


def gerar_grafico_eqm_epoca():
    totalEpocas = []
    for i in range(len(armazenaEQM)):
        totalEpocas.append(i)

    plt.plot(totalEpocas, armazenaEQM)
    plt.title('Grafico Eqm X Épocas')
    plt.xlabel('Épocas')
    plt.ylabel('Eqm')
    plt.show()


def gerar_grafico_obtido_esperado():
    tempos = []
    for i in range(len(resultados_testes)):
        tempos.append(i+len(amostras)+TIME_DELAY)

    esperados = []
    for i in range(len(conjuntoTesteResultados)):
        esperados.append(conjuntoTesteResultados[i][0])

    obtidos = []
    for i in range(len(resultados_testes)):
        obtidos.append(resultados_testes[i][0])

    fig, ax = plt.subplots()
    ax.plot(tempos, esperados, label ='Resultado Esperado')
    ax.plot(tempos, obtidos, label = 'Resultado Obtido')
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))

    plt.title('Resultado X Tempo')
    plt.xlabel('Tempo')
    plt.ylabel('Resultado')
    plt.legend()
    plt.show()


def exibir_total_epocas(total_epocas):
    print('\nTotal de Épocas: %i'% total_epocas)


def imprimir_tempo_processamento(start_time,end_time):
    print('\nTempo de treinamento: {}'.format(end_time - start_time))


def calcular_variancia():

    soma = 0

    for i in range(len(resultados_testes)):
        soma += sum(resultados_testes[i])

    media = soma / len(resultados_testes)

    soma = 0
    for i in range(len(resultados_testes)):
        soma += math.pow((resultados_testes[i][0] - media), 2)

    variancia = (soma / len(resultados_testes))

    print('Variância: %f' % variancia)


def calcular_erro_relativo():

    erros_absolutos = []

    for i in range(len(conjuntoTesteResultados)):
        erros_absolutos.append(abs(conjuntoTesteResultados[i][0] - resultados_testes[i][0]))

    erro_relativo = sum(erros_absolutos) / len(conjuntoTesteResultados)
    print('Erro Relativo: %f '% erro_relativo)


def exibirEQM():
    print('Erro Quadratico Médio: %f'% armazenaEQM[len(armazenaEQM)-1])


def main():

    carregar_dados()
    gerar_pesos_iniciais(pesos_camada_oculta, QTD_NEURONIOS_CAMADA_OCULTA, TIME_DELAY, False)
    gerar_pesos_iniciais(pesos_camada_saida, QTD_NEURONIOS_CAMADA_SAIDA, QTD_NEURONIOS_CAMADA_OCULTA + 1, False)
    gerar_pesos_iniciais(pesos_camada_oculta_anterior, QTD_NEURONIOS_CAMADA_OCULTA, TIME_DELAY, True)
    gerar_pesos_iniciais(pesos_camada_saida_anterior, QTD_NEURONIOS_CAMADA_SAIDA, QTD_NEURONIOS_CAMADA_OCULTA + 1, True)

    imprimir_pesos('Pesos Iniciais Camada Entrada', pesos_camada_oculta, 'ENTRADA')
    imprimir_pesos('Pesos Iniciais Camada Saída', pesos_camada_saida, 'SAIDA')

    start_time = datetime.now()
    epocas = treinar()
    end_time = datetime.now()

    imprimir_pesos('Pesos Finais Camada Entrada', pesos_camada_oculta, 'ENTRADA')
    imprimir_pesos('Pesos Finais Camada Saída', pesos_camada_saida, 'SAIDA')
    imprimir_tempo_processamento(start_time, end_time)
    exibir_total_epocas(epocas)

    testar()

    calcular_variancia()
    calcular_erro_relativo()
    exibirEQM()

    imprimir_resultados()
    gerar_grafico_eqm_epoca()
    gerar_grafico_obtido_esperado()

if __name__ == "__main__":
    main()
