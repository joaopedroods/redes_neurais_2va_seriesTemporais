# %% [secao] Protocolo: todas as decisões fixas do projeto
"""
rn_utils.py: código comum do projeto de Redes Neurais.

ATENÇÃO: este arquivo é gerado pelo notebook 02_codigo_comum.ipynb.
Não edite à mão. Edite o notebook 02 e rode-o de novo.
"""
import copy
import glob
import json
import math
import os
import random
import time
import zlib

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

HORIZONTE = 96   # passos previstos (H)
L_MAX = 720      # maior janela testada; fixa o início dos alvos de treino

DATASETS = {
    # periodo = passos em 1 dia (usado pelo seasonal naive)
    'ETTh1':   {'periodo': 24,  'lookbacks': [96, 192, 336, 512, 720], 'seeds': [0, 1, 2]},
    'Weather': {'periodo': 144, 'lookbacks': [96, 336, 720],           'seeds': [0, 1, 2]},
}

# Orçamento de treino: IGUAL para todos os modelos
TREINO = {'batch': 64, 'max_epocas': 30, 'paciencia': 5, 'clip_grad': 1.0}
LRS_CANDIDATOS = [1e-3, 3e-4, 1e-4]   # cada modelo escolhe 1 lr pela validação
L_AJUSTE = 336                         # janela usada para escolher o lr

# Testes de embaralhamento (só na avaliação; o modelo não é retreinado)
PERTURBACOES = ['embaralhar_tudo', 'embaralhar_antigo', 'embaralhar_recente', 'embaralhar_blocos']
FRACAO_RECENTE = 0.25   # "recente" = últimos 25% da janela
TAM_BLOCO = 16          # tamanho do bloco (todas as janelas são múltiplas de 16)
REPETICOES = 3          # permutações diferentes por teste; reporta-se a média

BASELINES = ['naive', 'sazonal', 'media_janela']

DISPOSITIVO = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PROJETO = os.path.abspath('.')


def configurar(projeto):
    """Define a pasta do projeto e cria as subpastas de saída."""
    global PROJETO
    PROJETO = os.path.abspath(projeto)
    for p in ['data/processed', 'resultados', 'checkpoints', 'figuras']:
        os.makedirs(os.path.join(PROJETO, p), exist_ok=True)
    print(f'Projeto: {PROJETO} | dispositivo: {DISPOSITIVO}')


def caminho(*partes):
    return os.path.join(PROJETO, *partes)


def fixar_semente(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# %% [secao] Dados: janelas com os MESMOS alvos para qualquer tamanho de janela
class Base:
    """Série univariada (OT) já limpa e normalizada pelo notebook 01_dados.

    Uma amostra é identificada por t, o índice do primeiro passo previsto:
      entrada = y[t-L : t]      alvo = y[t : t+H]
    Os índices t de cada split não dependem de L, então todas as janelas
    (96 a 720) são avaliadas exatamente nos mesmos pontos.
    """

    def __init__(self, nome):
        self.nome = nome
        df = pd.read_csv(caminho('data', 'processed', f'{nome}.csv'))
        with open(caminho('data', 'processed', f'{nome}_meta.json'), encoding='utf-8') as f:
            self.meta = json.load(f)
        self.datas = pd.to_datetime(df['date'])
        self.y = df['y'].to_numpy(np.float32)
        self.imputado = df['imputado'].to_numpy(bool)
        self.suspeito = df['suspeito'].to_numpy(bool)
        self.periodo = DATASETS[nome]['periodo']
        self.fim_treino = self.meta['fim_treino']
        self.fim_val = self.meta['fim_val']
        self.fim_teste = self.meta['fim_teste']
        self.y_t = torch.tensor(self.y, device=DISPOSITIVO)

    def _toca(self, mascara, t):
        """True para cada t cujo alvo [t, t+H) contém algum ponto marcado."""
        acum = np.concatenate([[0], np.cumsum(mascara)])
        return (acum[t + HORIZONTE] - acum[t]) > 0

    def indices_alvo(self, split):
        inicio, fim = {
            'treino': (L_MAX, self.fim_treino),       # começa em L_MAX: mesmo treino p/ todo L
            'val':    (self.fim_treino, self.fim_val),
            'teste':  (self.fim_val, self.fim_teste),
        }[split]
        t = np.arange(inicio, fim - HORIZONTE + 1)
        if split != 'teste':
            # nunca treinar/validar prevendo valores que nós mesmos imputamos
            t = t[~self._toca(self.imputado, t)]
        return t

    def alvo_suspeito(self, t):
        return self._toca(self.suspeito | self.imputado, t)

    def janelas(self, t, L):
        """Devolve X [B, L, 1] e Y [B, H, 1] no dispositivo."""
        t = torch.as_tensor(np.asarray(t), device=DISPOSITIVO).long()
        X = self.y_t[t[:, None] + torch.arange(-L, 0, device=DISPOSITIVO)]
        Y = self.y_t[t[:, None] + torch.arange(HORIZONTE, device=DISPOSITIVO)]
        return X[..., None], Y[..., None]


_CACHE = {}


def carregar_base(nome):
    if nome not in _CACHE:
        _CACHE[nome] = Base(nome)
    return _CACHE[nome]


# %% [secao] Perturbações: embaralhamentos idênticos para todos os modelos
def semente_perturbacao(ds, L, tipo, rep):
    """Semente fixa por (base, janela, teste, repetição).

    Não depende do modelo nem da seed de treino: todos os modelos são
    avaliados com exatamente as mesmas permutações.
    """
    return zlib.crc32(f'{ds}|{L}|{tipo}|{rep}'.encode())


def gerar_permutacao(tipo, n, L, semente):
    """Matriz [n, L] de índices; a linha i reordena a janela da amostra i.

    embaralhar_tudo     embaralha os L passos
    embaralhar_antigo   mantém os últimos 25% no lugar e embaralha o resto
    embaralhar_recente  embaralha só os últimos 25%
    embaralhar_blocos   troca a ordem de blocos de 16 passos, mantendo a ordem dentro de cada bloco
    """
    rng = np.random.default_rng(semente)
    idx = np.tile(np.arange(L), (n, 1))
    k = int(round(L * FRACAO_RECENTE))
    if tipo == 'embaralhar_tudo':
        idx = rng.permuted(idx, axis=1)
    elif tipo == 'embaralhar_antigo':
        idx[:, :L - k] = rng.permuted(idx[:, :L - k], axis=1)
    elif tipo == 'embaralhar_recente':
        idx[:, L - k:] = rng.permuted(idx[:, L - k:], axis=1)
    elif tipo == 'embaralhar_blocos':
        nb = L // TAM_BLOCO
        ordem = rng.permuted(np.tile(np.arange(nb), (n, 1)), axis=1)
        idx = (ordem[:, :, None] * TAM_BLOCO + np.arange(TAM_BLOCO)).reshape(n, L)
    else:
        raise ValueError(f'perturbação desconhecida: {tipo}')
    return idx


# %% [secao] Previsão, baselines e métricas
@torch.no_grad()
def prever(modelo, base, t, L, perm=None, lote=512):
    """Previsões do modelo para os alvos t, com a janela opcionalmente permutada."""
    modelo.eval()
    saidas = []
    for i in range(0, len(t), lote):
        X, _ = base.janelas(t[i:i + lote], L)
        if perm is not None:
            p = torch.as_tensor(perm[i:i + lote], device=DISPOSITIVO).long()
            X = torch.gather(X, 1, p[..., None].expand_as(X))
        saidas.append(modelo(X))
    return torch.cat(saidas)


@torch.no_grad()
def prever_baseline(nome, base, t, L):
    tt = torch.as_tensor(np.asarray(t), device=DISPOSITIVO).long()
    H = HORIZONTE
    if nome == 'naive':            # repete o último valor
        return base.y_t[tt - 1][:, None, None].expand(-1, H, 1)
    if nome == 'sazonal':          # repete o último dia
        P = base.periodo
        idx = tt[:, None] - P + (torch.arange(H, device=DISPOSITIVO) % P)
        return base.y_t[idx][..., None]
    if nome == 'media_janela':     # o que um modelo com RevIN prevê se não aprender nada
        X, _ = base.janelas(t, L)
        return X.mean(1, keepdim=True).expand(-1, H, -1)
    raise ValueError(nome)


def erros_por_amostra(pred, Y):
    d = pred - Y
    return (d ** 2).mean(dim=(1, 2)).cpu().numpy(), d.abs().mean(dim=(1, 2)).cpu().numpy()


def resumir(mse, mae, suspeito):
    return {
        'mse': float(mse.mean()), 'mae': float(mae.mean()),
        'mse_sem_suspeitos': float(mse[~suspeito].mean()),
        'mae_sem_suspeitos': float(mae[~suspeito].mean()),
    }


def avaliar_completo(prever_fn, base, L, perturbar=True):
    """Avalia no teste: original e, se perturbar=True, os 4 embaralhamentos."""
    t = base.indices_alvo('teste')
    _, Y = base.janelas(t, L)
    susp = base.alvo_suspeito(t)
    mse0, mae0 = erros_por_amostra(prever_fn(t, None), Y)
    res = {'original': resumir(mse0, mae0, susp)}
    erros = {'t': t, 'original_mse': mse0, 'original_mae': mae0}
    if perturbar:
        for tipo in PERTURBACOES:
            reps, acum = [], np.zeros_like(mse0)
            for r in range(REPETICOES):
                perm = gerar_permutacao(tipo, len(t), L, semente_perturbacao(base.nome, L, tipo, r))
                m, a = erros_por_amostra(prever_fn(t, perm), Y)
                reps.append(resumir(m, a, susp))
                acum += m
            media = {k: float(np.mean([x[k] for x in reps])) for k in reps[0]}
            media['mse_std_reps'] = float(np.std([x['mse'] for x in reps]))
            media['razao_mse'] = media['mse'] / res['original']['mse']
            res[tipo] = media
            erros[f'{tipo}_mse'] = acum / REPETICOES
    return res, erros


# %% [secao] Treino: mesmo laço, mesma normalização e mesmo orçamento para todos
class ComRevIN(nn.Module):
    """Normaliza cada janela pela própria média e desvio, aplica o modelo e desfaz na saída.

    É aplicado a TODOS os modelos, para que a normalização não seja uma
    vantagem de um só (o PatchTST original já usa; os outros não).
    Média e desvio não mudam quando a janela é embaralhada, então isso não
    vaza informação de ordem.
    """

    def __init__(self, modelo, eps=1e-5):
        super().__init__()
        self.modelo = modelo
        self.eps = eps

    def forward(self, x):
        media = x.mean(1, keepdim=True)
        desvio = torch.sqrt(x.var(1, keepdim=True, unbiased=False) + self.eps)
        return self.modelo((x - media) / desvio) * desvio + media


def _salvar_parcial(arq, estado):
    """Mesmo cuidado do JSON: grava num temporário e renomeia."""
    tmp = arq + '.tmp'
    torch.save(estado, tmp)
    os.replace(tmp, arq)


def treinar(modelo, base, L, lr, seed, verbose=True, arq_parcial=None):
    """Treina com early stopping. Com `arq_parcial`, salva o progresso a cada época e,
    se o arquivo já existir (execução interrompida), continua da última época salva,
    com o mesmo estado de pesos, otimizador e geradores aleatórios."""
    cfg = TREINO
    t_tr = base.indices_alvo('treino')
    t_va = base.indices_alvo('val')
    _, Y_va = base.janelas(t_va, L)
    opt = torch.optim.Adam(modelo.parameters(), lr=lr)
    perda = nn.MSELoss()
    rng = np.random.default_rng(seed)
    melhor, melhor_estado, melhor_ep, sem_melhora = math.inf, None, 0, 0
    historico, gasto, ep0, parar = [], 0.0, 1, False
    if arq_parcial and os.path.exists(arq_parcial):
        p = torch.load(arq_parcial, map_location='cpu', weights_only=False)
        modelo.load_state_dict(p['modelo'])
        opt.load_state_dict(p['otimizador'])
        rng.bit_generator.state = p['rng']
        torch.set_rng_state(p['torch_rng'])
        if torch.cuda.is_available() and p['cuda_rng'] is not None:
            torch.cuda.set_rng_state_all(p['cuda_rng'])
        melhor, melhor_estado, melhor_ep, sem_melhora = p['melhor'], p['melhor_estado'], p['melhor_ep'], p['sem_melhora']
        historico, gasto, ep0, parar = p['historico'], p['tempo_s'], p['epoca'] + 1, p['parar']
        print(f'  ↻ retomando: época {p["epoca"]} já estava salva')
    t0 = time.time()
    for ep in range(ep0, cfg['max_epocas'] + 1):
        if parar:
            break
        modelo.train()
        ordem = rng.permutation(t_tr)
        soma, n = 0.0, 0
        for i in range(0, len(ordem), cfg['batch']):
            lote = ordem[i:i + cfg['batch']]
            if len(lote) < 2:      # evita erro de BatchNorm com lote de 1
                continue
            X, Y = base.janelas(lote, L)
            opt.zero_grad()
            l = perda(modelo(X), Y)
            l.backward()
            nn.utils.clip_grad_norm_(modelo.parameters(), cfg['clip_grad'])
            opt.step()
            soma += l.item() * len(lote)
            n += len(lote)
        val = float(((prever(modelo, base, t_va, L) - Y_va) ** 2).mean())
        historico.append({'epoca': ep, 'mse_treino': soma / n, 'mse_val': val})
        if verbose:
            print(f'  época {ep:2d}  treino {soma / n:.4f}  val {val:.4f}')
        if not math.isfinite(val):
            parar = True
        elif val < melhor - 1e-6:
            melhor, melhor_ep, sem_melhora = val, ep, 0
            melhor_estado = copy.deepcopy(modelo.state_dict())
        else:
            sem_melhora += 1
            parar = sem_melhora >= cfg['paciencia']
        if arq_parcial:
            _salvar_parcial(arq_parcial, {
                'epoca': ep, 'parar': parar, 'modelo': modelo.state_dict(), 'otimizador': opt.state_dict(),
                'rng': rng.bit_generator.state, 'torch_rng': torch.get_rng_state(),
                'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                'melhor': melhor, 'melhor_estado': melhor_estado, 'melhor_ep': melhor_ep,
                'sem_melhora': sem_melhora, 'historico': historico, 'tempo_s': gasto + time.time() - t0})
    if melhor_estado is not None:
        modelo.load_state_dict(melhor_estado)
    return {'historico': historico, 'melhor_epoca': melhor_ep, 'mse_val': melhor,
            'epocas_rodadas': len(historico), 'tempo_treino_s': gasto + time.time() - t0}


# %% [secao] Execução: grade retomável (pula o que já tem resultado salvo)
def _salvar_json(arq, dados):
    """Grava em arquivo temporário e renomeia: uma interrupção não deixa JSON pela metade."""
    tmp = arq + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(dados, f, indent=2, ensure_ascii=False)
    os.replace(tmp, arq)


def id_execucao(ds, modelo, L, lr, seed):
    return f'{ds}_{modelo}_L{L}_H{HORIZONTE}_lr{lr:g}_s{seed}'


def executar(nome, criar_modelo, ds, L, lr, seed, forcar=False, salvar_checkpoint=True, verbose=False):
    """Treina e avalia uma configuração. Se o resultado já existe, só o carrega.

    criar_modelo(L, H) deve devolver um nn.Module que recebe [B, L, 1] e devolve [B, H, 1].
    """
    rid = id_execucao(ds, nome, L, lr, seed)
    pasta = caminho('resultados', ds, nome)
    arq = os.path.join(pasta, rid + '.json')
    if os.path.exists(arq) and not forcar:
        with open(arq, encoding='utf-8') as f:
            return json.load(f)
    os.makedirs(pasta, exist_ok=True)
    pasta_ck = caminho('checkpoints', ds, nome)
    os.makedirs(pasta_ck, exist_ok=True)
    arq_parcial = os.path.join(pasta_ck, rid + '_parcial.pt')   # progresso época a época (apagado no fim)
    if forcar and os.path.exists(arq_parcial):
        os.remove(arq_parcial)
    base = carregar_base(ds)
    fixar_semente(seed)
    modelo = ComRevIN(criar_modelo(L, HORIZONTE)).to(DISPOSITIVO)
    n_param = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    print(f'▶ {rid}  ({n_param:,} parâmetros)')
    info_treino = treinar(modelo, base, L, lr, seed, verbose=verbose, arq_parcial=arq_parcial)
    res, erros = avaliar_completo(lambda t, perm: prever(modelo, base, t, L, perm), base, L)
    info = {'id': rid, 'tipo': 'rede', 'dataset': ds, 'modelo': nome, 'L': L, 'H': HORIZONTE,
            'lr': lr, 'seed': seed, 'n_parametros': n_param, **info_treino, 'teste': res,
            'dispositivo': str(DISPOSITIVO), 'treino_cfg': TREINO}
    np.savez_compressed(os.path.join(pasta, rid + '.npz'), **erros)
    if salvar_checkpoint:
        torch.save(modelo.state_dict(), os.path.join(pasta_ck, rid + '.pt'))
    _salvar_json(arq, info)   # por último: JSON existir = execução completa
    if os.path.exists(arq_parcial):
        os.remove(arq_parcial)
    o = res['original']
    print(f'  ✓ val {info["mse_val"]:.4f} | teste MSE {o["mse"]:.4f} MAE {o["mae"]:.4f} | '
          f'embaralhar_tudo ×{res["embaralhar_tudo"]["razao_mse"]:.2f} | '
          f'{info["epocas_rodadas"]} épocas, {info["tempo_treino_s"]:.0f}s')
    return info


def escolher_lr(nome, criar_modelo, ds):
    """Mesmo orçamento de ajuste para todos: 3 lrs, janela 336, seed 0, escolha pela validação."""
    arq = caminho('resultados', ds, nome, 'lr_escolhido.json')
    if os.path.exists(arq):
        with open(arq, encoding='utf-8') as f:
            salvo = json.load(f)
        print(f'lr escolhido para {nome} em {ds}: {salvo["lr"]:g}  '
              f'(val: { {k: round(v, 4) for k, v in salvo["mse_val_por_lr"].items()} })')
        return salvo['lr']
    vals = {lr: executar(nome, criar_modelo, ds, L_AJUSTE, lr, seed=0)['mse_val'] for lr in LRS_CANDIDATOS}
    lr = min(vals, key=vals.get)
    _salvar_json(arq, {'lr': lr, 'mse_val_por_lr': {f'{k:g}': v for k, v in vals.items()}})
    print(f'lr escolhido para {nome} em {ds}: {lr:g}  (val: { {f"{k:g}": round(v, 4) for k, v in vals.items()} })')
    return lr


def rodar_grade(nome, criar_modelo, datasets=('ETTh1', 'Weather'), seeds=None):
    """Roda tudo de um modelo: busca de lr e depois todas as janelas e seeds.

    Execuções já salvas são apenas recarregadas (e listadas); uma execução interrompida
    continua da última época salva. `seeds` (opcional) restringe as seeds executadas;
    a busca de lr usa a seed 0, então precisa existir antes de rodar só as seeds 1 ou 2.
    """
    for ds in datasets:
        alvo = DATASETS[ds]['seeds'] if seeds is None else [s for s in DATASETS[ds]['seeds'] if s in seeds]
        if seeds is not None and 0 not in alvo and \
                not os.path.exists(caminho('resultados', ds, nome, 'lr_escolhido.json')):
            raise RuntimeError(f'{nome} em {ds}: a busca de learning rate (feita com a seed 0) ainda não foi feita.')
        lr = escolher_lr(nome, criar_modelo, ds)
        for L in DATASETS[ds]['lookbacks']:
            for seed in alvo:
                rid = id_execucao(ds, nome, L, lr, seed)
                ja_salvo = os.path.exists(caminho('resultados', ds, nome, rid + '.json'))
                info = executar(nome, criar_modelo, ds, L, lr, seed)
                if ja_salvo:
                    o = info['teste']['original']
                    print(f'  · {rid} (salvo) | val {info["mse_val"]:.4f} | teste MSE {o["mse"]:.4f} '
                          f'MAE {o["mae"]:.4f} | embaralhar_tudo ×{info["teste"]["embaralhar_tudo"]["razao_mse"]:.2f} | '
                          f'{info["epocas_rodadas"]} épocas, {info["tempo_treino_s"]:.0f}s ({info.get("dispositivo", "?")})')
    print(f'Grade de {nome} completa' + ('.' if seeds is None else f' para as seeds {list(seeds)}.'))


def rodar_baselines(datasets=('ETTh1', 'Weather')):
    for ds in datasets:
        base = carregar_base(ds)
        for L in DATASETS[ds]['lookbacks']:
            for b in BASELINES:
                pasta = caminho('resultados', ds, b)
                rid = f'{ds}_{b}_L{L}_H{HORIZONTE}'
                arq = os.path.join(pasta, rid + '.json')
                if os.path.exists(arq):
                    continue
                os.makedirs(pasta, exist_ok=True)
                res, erros = avaliar_completo(lambda t, perm: prever_baseline(b, base, t, L),
                                              base, L, perturbar=False)
                np.savez_compressed(os.path.join(pasta, rid + '.npz'), **erros)
                _salvar_json(arq, {'id': rid, 'tipo': 'baseline', 'dataset': ds, 'modelo': b,
                                   'L': L, 'H': HORIZONTE, 'lr': None, 'seed': None, 'teste': res})
    print('Baselines prontos.')


# %% [secao] Resultados: tabela única com tudo que já rodou
def carregar_resultados():
    """Uma linha por execução. Das redes, mantém só o lr escolhido pela validação."""
    escolhidos = {}
    for arq in glob.glob(caminho('resultados', '*', '*', 'lr_escolhido.json')):
        ds, mod = arq.replace('\\', '/').split('/')[-3:-1]
        with open(arq, encoding='utf-8') as f:
            escolhidos[(ds, mod)] = json.load(f)['lr']
    linhas = []
    for arq in glob.glob(caminho('resultados', '*', '*', '*.json')):
        if arq.endswith('lr_escolhido.json'):
            continue
        with open(arq, encoding='utf-8') as f:
            r = json.load(f)
        if r['tipo'] == 'rede' and r['lr'] != escolhidos.get((r['dataset'], r['modelo'])):
            continue
        linha = {k: r.get(k) for k in ['id', 'dataset', 'modelo', 'tipo', 'L', 'lr', 'seed', 'n_parametros',
                                       'melhor_epoca', 'epocas_rodadas', 'tempo_treino_s', 'mse_val', 'dispositivo']}
        for cond, m in r['teste'].items():
            linha[f'{cond}_mse'] = m['mse']
            linha[f'{cond}_mae'] = m['mae']
            linha[f'{cond}_mse_ss'] = m['mse_sem_suspeitos']
            if 'razao_mse' in m:
                linha[f'{cond}_razao'] = m['razao_mse']
        linhas.append(linha)
    df = pd.DataFrame(linhas)
    if len(df):
        df = df.sort_values(['dataset', 'tipo', 'modelo', 'L', 'seed'], na_position='first').reset_index(drop=True)
    return df
