"""Render the maintainable local architecture diagram without external tools."""
from pathlib import Path
from html import escape

NODES={
 'engine':(25,65,175,90,'Game engine','Grid rules / food RNG'),
 'env':(245,65,175,90,'RL environment','Rewards / termination'),
 'encoder':(465,65,175,90,'Observation encoder','11 / 13 / 14 / 16'),
 'agent':(685,65,175,90,'DQN agent','Greedy / exploration'),
 'action':(905,65,165,90,'Relative action','Straight / right / left'),
 'replay':(25,300,205,100,'Experience replay','Transitions / minibatches'),
 'online':(295,300,205,100,'Online network','Q(s,a) / action choice'),
 'target':(565,300,205,100,'Target network','Detached future values'),
 'update':(835,300,235,100,'Training updates','Huber loss / Adam / clip'),
 'checkpoint':(295,550,205,90,'Model checkpoints','Weights / config / counters'),
 'evaluation':(565,550,205,90,'Greedy evaluation','Fresh paired environment seeds'),
 'results':(835,550,235,90,'Experiment records','CSV / JSON / charts / report'),
}
LINKS=[('engine','env','implements rules'),('env','encoder','state'),('encoder','agent','float32 features'),
 ('agent','action','action values'),('action','env','env.step'),('env','replay','training transitions'),
 ('replay','online','sample batch'),('replay','target','next states'),('online','target','periodic copy / Double action selection'),
 ('online','update','selected Q(s,a)'),('target','update','bootstrap target'),('update','online','optimise online weights'),
 ('online','checkpoint','save'),('checkpoint','evaluation','load for inference'),('evaluation','env','same rules'),
 ('evaluation','results','outcomes'),('agent','online','inference uses online'),('replay','update','rewards / terminal mask')]

def build():
 root=Path(__file__).resolve().parent
 m=['flowchart LR']
 for key,(_,_,_,_,title,detail) in NODES.items():m.append(f'  {key}["{title}<br/>{detail}"]')
 for a,b,label in LINKS:m.append(f'  {a} -->|"{label}"| {b}')
 (root/'architecture.mmd').write_text('\n'.join(m)+'\n',encoding='utf-8')
 parts=['<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="730" viewBox="0 0 1100 730">',
 '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" fill="#476273"/></marker></defs>',
 '<rect width="1100" height="730" fill="#f2f6f9"/><g font-family="Arial,sans-serif" fill="#183147">',
 '<text x="25" y="30" font-size="24" font-weight="bold">SnakeAI / inference and learning</text>',
 '<rect x="12" y="265" width="1076" height="220" rx="10" fill="#dfebee"/>',
 '<text x="25" y="287" font-size="13">Training only</text>']
 def arrow(path,label,x,y,dashed=False):
  parts.append(f'<path d="{path}" fill="none" stroke="#476273" stroke-width="1.7" marker-end="url(#arrow)"'+(' stroke-dasharray="5 4"' if dashed else '')+'/>')
  if label:parts.append(f'<text x="{x}" y="{y}" font-size="11" fill="#38596b">{escape(label)}</text>')
 for a,b,label in LINKS[:4]:
  label={'implements rules':'rules','float32 features':'features','action values':'Q(s)'}.get(label,label)
  x,y,w,h,_,_=NODES[a];xx,yy,_,_,_,_=NODES[b];arrow(f'M{x+w},{y+h/2} L{xx},{yy+h/2}',label,x+w+3,y+h/2-9)
 arrow('M985,155 L985,210 L333,210 L333,155','env.step(action)',590,202)
 arrow('M730,155 L730,227 L450,227 L450,300','online inference',470,219)
 arrow('M275,155 L275,240 L128,240 L128,300','training transition',35,232)
 arrow('M230,345 L295,345','batch',244,334)
 arrow('M128,400 L128,449 L620,449 L620,400','next states',160,442)
 arrow('M500,335 L565,335','copy',516,324,True)
 arrow('M500,373 L565,373','Double a′',505,390)
 arrow('M770,345 L835,345','target',783,334)
 arrow('M400,300 L400,250 L950,250 L950,300','Q(s,a) for regression',650,242)
 arrow('M128,400 L128,428 L1020,428 L1020,400','rewards / terminal mask',785,420)
 arrow('M950,400 L950,475 L400,475 L400,400','Adam updates online weights only',585,468)
 arrow('M350,400 L350,550','save weights + metadata',355,525)
 arrow('M500,595 L565,595','load',518,583)
 arrow('M770,595 L835,595','results',779,583)
 arrow('M668,640 L668,686 L1090,686 L1090,45 L333,45 L333,65','evaluation reuses the same environment; no training updates',720,678,True)
 for key,(x,y,w,h,title,detail) in NODES.items():
  parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="7" fill="white" stroke="#9cb4c0"/>')
  parts.append(f'<text x="{x+12}" y="{y+32}" font-size="16" font-weight="bold">{escape(title)}</text>')
  words=[detail] if key=='encoder' else (['CSV / JSON','Charts / report'] if key=='results' else detail.split(' / '))
  for i,line in enumerate(words):parts.append(f'<text x="{x+12}" y="{y+55+i*15}" font-size="11" fill="#526b7b">{escape(line)}</text>')
 parts += ['<text x="25" y="715" font-size="12">Standard: target selects + evaluates. Double: online selects, target evaluates. Genuine terminal transitions do not bootstrap.</text>','</g></svg>']
 (root/'architecture.svg').write_text(''.join(parts),encoding='utf-8')

if __name__=='__main__':build()
