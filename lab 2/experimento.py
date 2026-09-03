import csv
import socket
import subprocess
import sys
import time
from pathlib import Path

import ray

from mlcliente import executar_cliente


NUM_CLIENTS = 5
SERVER_PORT = 50051
OUTPUT_DIR = Path(__file__).with_name('output')
REPORT_PATH = OUTPUT_DIR / 'relatorio_experimentos.csv'
MARKDOWN_PATH = OUTPUT_DIR / 'relatorio_experimentos.md'

BEHAVIOR_NAMES = {
    1: 'ruido em todas as features',
    2: 'ruido em uma feature',
    3: 'rotulos invertidos',
}


@ray.remote
class ClienteActor:
    def run(self, client_id, behavior_id):
        return executar_cliente(client_id, behavior_id)


def wait_for_server():
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            with socket.create_connection(('localhost', SERVER_PORT), timeout=0.2):
                return
        except OSError:
            time.sleep(0.1)
    raise RuntimeError('servidor não iniciou a tempo')


def run_scenario(behavior_id, malicious_count):
    server = subprocess.Popen(
        [sys.executable, 'mlservidor.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        wait_for_server()
        actors = [ClienteActor.remote() for _ in range(NUM_CLIENTS)]
        try:
            futures = [
                actor.run.remote(
                    client_id,
                    behavior_id if client_id < malicious_count else 0,
                )
                for client_id, actor in enumerate(actors)
            ]
            results = ray.get(futures)
        finally:
            for actor in actors:
                ray.kill(actor)

        accepted = [result['accepted'] for result in results]
        acc_with_detection = float(results[0]['acc_with_detection'])
        acc_without_detection = float(results[0]['acc_without_detection'])

        return {
            'tipo_bizantino': BEHAVIOR_NAMES[behavior_id],
            'behavior_id': behavior_id,
            'clientes_bizantinos': malicious_count,
            'clientes_aceitos': sum(accepted),
            'clientes_rejeitados': NUM_CLIENTS - sum(accepted),
            'acuracia_com_deteccao': acc_with_detection,
            'acuracia_sem_deteccao': acc_without_detection,
        }
    finally:
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
        time.sleep(0.2)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    fieldnames = [
        'tipo_bizantino',
        'behavior_id',
        'clientes_bizantinos',
        'clientes_aceitos',
        'clientes_rejeitados',
        'acuracia_com_deteccao',
        'acuracia_sem_deteccao',
    ]

    rows = []
    ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False)
    try:
        for behavior_id in BEHAVIOR_NAMES:
            for malicious_count in range(1, 4):
                print(
                    f'Executando: tipo={behavior_id}, '
                    f'clientes_bizantinos={malicious_count}'
                )
                row = run_scenario(behavior_id, malicious_count)
                rows.append(row)
                print(
                    f"  com detecção={row['acuracia_com_deteccao']:.4f}; "
                    f"sem detecção={row['acuracia_sem_deteccao']:.4f}; "
                    f"aceitos={row['clientes_aceitos']}"
                )
    finally:
        ray.shutdown()

    with REPORT_PATH.open('w', newline='', encoding='utf-8') as report:
        writer = csv.DictWriter(report, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f'Relatório salvo em: {REPORT_PATH}')
    write_markdown_report(rows)


def write_markdown_report(rows):
    lines = [
        '# Relatório de Experimentos Federados',
        '',
        '## Configuração',
        '',
        '- Dataset: `load_digits`, convertido em classificação binária (pares = 0, ímpares = 1).',
        '- Clientes por cenário: 5.',
        '- Cenários: 1, 2 e 3 clientes bizantinos para cada tipo de ataque.',
        '- Critério de aceitação: acurácia no teste maior que `0.5` (chute aleatório).',
        '- Agregação: média dos pesos dos modelos aceitos.',
        '',
        '## Resultados',
        '',
        '| Tipo de ataque | Bizantinos | Aceitos | Rejeitados | Com detecção | Sem detecção |',
        '|---|---:|---:|---:|---:|---:|',
    ]

    for row in rows:
        lines.append(
            f"| {row['tipo_bizantino']} | {row['clientes_bizantinos']} | "
            f"{row['clientes_aceitos']} | {row['clientes_rejeitados']} | "
            f"{float(row['acuracia_com_deteccao']):.4f} | "
            f"{float(row['acuracia_sem_deteccao']):.4f} |"
        )

    lines.extend([
        '',
        '## Observação',
        '',
        'A comparação mostra o desempenho do modelo final após a filtragem dos updates '
        'em relação à agregação de todos os modelos recebidos.',
        '',
    ])
    MARKDOWN_PATH.write_text('\n'.join(lines), encoding='utf-8')
    print(f'Relatório Markdown salvo em: {MARKDOWN_PATH}')


if __name__ == '__main__':
    main()
