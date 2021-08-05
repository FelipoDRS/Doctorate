# -*- coding: utf-8 -*-
"""
Created on Wed May 19 10:19:58 2021

@author: Felipo Soares
"""

from tkinter import * 
class Janela: 
    def __init__(self,toplevel):
        self.fr1 = Frame(toplevel)
        self.fr1.pack()
        self.botao1 = Button(self.fr1,text='Oi!') 
        self.botao1['background']='green'        
        self.botao1['font']=('Verdana','12','italic','bold')         
        self.botao1['height']=3         
        
        self.botao1.pack()         
        self.botao2 = Button(self.fr1,bg='red', font=('Times','16'))         
        self.botao2['text']='Tchau!'        
        self.botao2['fg']='yellow'        
        self.botao2['width']=12         
        self.botao2.pack() 

raiz=Tk() 
Janela(raiz) 
raiz.mainloop()

class Packing: 
    def __init__ (self,instancia_Tk):         
        self.container1=Frame(instancia_Tk)         
        self.container2=Frame(instancia_Tk)         
        self.container3=Frame(instancia_Tk)         
        self.container1.pack(side=BOTTOM)         
        self.container3.pack(side=TOP)         
        self.container2.pack()        
        
        Button(self.container1,text='B1').pack()         
        Button(self.container2,text='B2').pack(side=LEFT)         
        Button(self.container2,text='B3').pack(side=LEFT)         
        self.b4=Button(self.container3,text='B4')         
        self.b5=Button(self.container3,text='B5')         
        self.b6=Button(self.container3,text='B6')         
        self.b6.pack(side=RIGHT)         
        self.b4.pack(side=RIGHT)         
        self.b5.pack(side=RIGHT)         
        
raiz=Tk() 
Packing(raiz) 
raiz.mainloop()

class Janela:  
   def __init__ (self,toplevel):         
       self.frame=Frame(toplevel)         
       self.frame.pack()
       self.frame2=Frame(toplevel)         
       self.frame2.pack()          
       
       self.titulo=Label(self.frame,text='VIDENTE2005',               
                         font=('Verdana','13','bold'))         
       self.titulo.pack()
       
       self.msg=Label(self.frame,width=40,height=6,                          
                      text = 'Adivinho o evento ocorrido!')         
       self.msg.focus_force()         
       self.msg.pack() 
       
       self.b01=Button(self.frame2,text='Botão 1')         
       self.b01['padx'], self.b01['pady'] = 10, 5         
       self.b01['bg']='deepskyblue'        
       self.b01.bind("<Return>",self.keypress01)         
       self.b01.bind("<Any-Button>",self.button01)         
       self.b01.bind("<FocusIn>",self.fin01)         
       self.b01.bind("<FocusOut>",self.fout01)         
       self.b01['relief']=RIDGE         
       self.b01.pack(side=LEFT)
       self.b02=Button(self.frame2,text='Botão 2')         
       self.b02['padx'], self.b02['pady'] = 5, 15         
       self.b02['bg']='deepskyblue'        
       self.b02.bind("<Return>",self.keypress02)         
       self.b02.bind("<Any-Button>",self.button02)         
       self.b02.bind("<FocusIn>",self.fin02)         
       self.b02.bind("<FocusOut>",self.fout02)         
       self.b02['relief']=RIDGE         
       self.b02.pack(side=LEFT)
        
   def keypress01(self,event): self.msg['text']='ENTER sobre o Botão 1' 
   def keypress02(self,event): self.msg['text']='ENTER sobre o Botão 2' 
   def button01(self,event): self.msg['text']='Clique sobre o Botão 1' 
   def button02(self,event): self.msg['text']='Clique sobre o Botão 2' 
   def fin01(self,event): self.b01['relief']=FLAT 
   def fout01(self,event): self.b01['relief']=RIDGE 
   def fin02(self,event): self.b02['relief']=FLAT 
   def fout02(self,event): self.b02['relief']=GROOVE  





           
raiz=Tk() 
Janela(raiz) 
raiz.mainloop()
# -*- coding: cp1252 -*- 
           
           
class Janela: 
    def __init__(self,toplevel):
         self.frame=Frame(toplevel)
         self.frame.pack()
         self.b1=Button(self.frame)
         self.b1.bind("<Button-1>", self.press_b1)
         self.b1.bind("<ButtonRelease>", self.release_b1)
         self.b1['text'] = 'Clique em mim!'        
         self.b1['width'], self.b1['bg'] = 20, 'brown'        
         self.b1['fg']='yellow'        
         self.b1.pack(side=LEFT)
         self.b2=Button(self.frame)
         self.b2['width'], self.b2['bg'] = 20, 'brown'        
         self.b2['fg']='yellow'        
         self.b2.pack(side=LEFT)
         self.b3=Button(self.frame, command=self.click_b3)
         self.b3['width'], self.b3['bg'] = 20, 'brown'        
         self.b3['fg']='yellow'        
         self.b3.pack(side=LEFT) 
         
    def press_b1(self,event):
         self.b1['text']=''        
         self.b2['text']='Errou! Estou aqui!'
    def release_b1(self,event):
         self.b2['text']=''        
         self.b3['text']='OOOpa! Mudei de novo!'
    def click_b3(self):
         self.b3['text']='Ok... Você me pegou...'   
         
         
         
class Passwords: 
    def __init__(self,toplevel):
         self.frame1=Frame(toplevel)
         self.frame1.pack()
         self.frame2=Frame(toplevel,pady=10)
         self.frame2.pack()
         self.frame3=Frame(toplevel,pady=10)
         self.frame3.pack()
         self.frame4=Frame(toplevel,pady=10)
         self.frame4.pack()
         Label(self.frame1,text='PASSWORDS', fg='darkblue',
               font=('Verdana','14','bold'), height=3).pack()
         fonte1=('Verdana','10','bold')
         
         Label(self.frame2,text='Nome: ',
               font=fonte1,width=8).pack(side=LEFT)
         self.nome=Entry(self.frame2,width=10,show='*',
                          font=fonte1)
         #self.nome.focus_force() # Para o foco começar neste campo        self.nome.pack(side=LEFT)
         self.nome.pack(side=LEFT)

         Label(self.frame3,text='Senha: ',
               font=fonte1,width=8).pack(side=LEFT)
         self.senha=Entry(self.frame3,width=10,         
                 font=fonte1)
         self.senha.pack(side=LEFT)
         self.confere=Button(self.frame4, font=fonte1, text='Conferir',
                             bg='pink', command=self.conferir)
         self.confere.pack()
         self.msg=Label(self.frame4,font=fonte1,height=3,text='AGUARDANDO...')         
         self.msg.pack()

    def conferir(self):
         NOME=self.nome.get()
         SENHA=self.senha.get() 
         if NOME == SENHA:
             self.msg['text']='ACESSOPERMITIDO'
             self.msg['fg']='darkgreen'
         else:
             self.msg['text']='ACESSONEGADO'
             self.msg['fg']='red'
             self.nome.focus_force() 
             
raiz=Tk() 
Passwords(raiz) 
raiz.mainloop()


class Kanvas: 
    def __init__(self,raiz):
         self.canvas1 = Canvas(raiz, width=100, height=200,
           cursor='X_cursor', bd=5,
           bg='dodgerblue')
         self.canvas1.pack(side=LEFT)
         self.canvas2 = Canvas(raiz, width=100, height=200,
           cursor='dot', bd=5,
           bg='purple')
         self.canvas2.pack(side=LEFT)
         
raiz=Tk() 
Kanvas(raiz) 
raiz.mainloop()

class Linhas: 
    def __init__(self,raiz):
         self.canvas = Canvas(raiz, width=400, height=400,
         
         
           cursor='watch', bd=5)
         self.canvas.pack()
         self.frame=Frame(raiz)
         self.frame.pack()
         self.last=[200,200]
         configs={'fg':'darkblue', 'bg':'ghostwhite', 'relief':GROOVE, 'width':11,'font':('Verdana','8','bold')}
         self.b1=Button(self.frame, configs,
         
               text='Esquerda', command=self.left)
         self.b1.pack(side=LEFT)
         self.b2=Button(self.frame, configs,
         
               text='Paracima', command=self.up)
         self.b2.pack(side=LEFT)
         self.b3=Button(self.frame, configs,
         
               text='Parabaixo', command=self.down)
         self.b3.pack(side=LEFT)
         self.b4=Button(self.frame, configs,
         
               text='Direita', command=self.right)
         self.b4.pack(side=LEFT)
    def left(self): # Desenha um segmento para a esquerda        
             x, y = self.last[0]-10, self.last[1]
             self.canvas.create_line(self.last, x, y, fill='red',width=20)
             self.last=[x,y] 
    def up(self): # Desenha um segmento para cima        
             x, y = self.last[0], self.last[1]-10
             self.canvas.create_line(self.last, x, y, fill='yellow',width=20)
             self.last=[x,y] 
    def down(self): # Desenha um segmento para baixo        
             x, y = self.last[0], self.last[1]+10
             self.canvas.create_line(self.last, x, y, fill='blue',width=20)
             self.last=[x,y] 
    def right(self): # Desenha um segmento para a direita        
             x, y = self.last[0]+10, self.last[1]         
             self.canvas.create_line(self.last, x, y, fill='purple',width=20) 
             self.last=[x,y]
             
             
raiz=Tk() 
Linhas(raiz) 
raiz.mainloop()

class SPFC: 
    def __init__(self,raiz):         
        self.canvas=Canvas(raiz, width=200, height=200, bg='dodgerblue')         
        self.canvas.pack()         
        altura = 200 # Altura do canvas        
        pol=self.canvas.create_polygon         
        ret=self.canvas.create_rectangle         
        pol(100, altura-10,
             10, altura-145,
             10, altura-190,
             190, altura-190,
             190, altura-145,
             100, altura-10, fill='white')         
        ret(15, altura-150, 185, altura-185, fill='black')         
        pol(20, altura-140,
             95, altura-140,
             95, altura-30,
             20, altura-140, fill='red')         
        pol(105, altura-30,
             105, altura-140,
             180, altura-140,
             105, altura-30, fill='black')         
        self.canvas.create_text(100, altura-167.5, text='S  P  F  C',
             
                    font=('Arial','24','bold'),
             
                    anchor=CENTER, fill='white')
        
raiz=Tk() 
SPFC(raiz) 
raiz.mainloop()


class Fatias: 
    def __init__(self,raiz):
         self.canvas=Canvas(raiz, width=200, height=200)
         self.canvas.pack()
         self.frame=Frame(raiz)
         self.frame.pack()
         self.altura = 200 # Altura do canvas  
         self.canvas.create_oval(25,  self.altura-25,
         175, self.altura-175,
               fill='deepskyblue', outline='darkblue')
         
         fonte=('Comic Sans MS', '14', 'bold')
         Label(self.frame, text='Fatia: ',
               font=fonte, fg='blue').pack(side=LEFT)
         self.porcentagem=Entry(self.frame, fg='red',
         font=fonte, width=5)        
         self.porcentagem.focus_force()
         self.porcentagem.pack(side=LEFT)
         Label(self.frame, text='%',
               font=fonte, fg='blue').pack(side=LEFT)
         self.botao=Button(self.frame, text='Desenhar',
         
         
         command=self.cortar, font=fonte,
         
         
         fg='darkblue', bg='deepskyblue')
         self.botao.pack(side=LEFT) 
    def cortar(self):
         arco=self.canvas.create_arc
         fatia=float(self.porcentagem.get())*359.9/100.
         arco(25, self.altura-25,
              175, self.altura-175, 

         fill='yellow', outline='red',
              extent=fatia,style='chord')
         self.porcentagem.focus_force() 


instancia=Tk() 
Fatias(instancia) 
instancia.mainloop()

from time import localtime 

class Horas: 
    def __init__(self,raiz):
         self.canvas=Canvas(raiz, width=200, height=100)
         self.canvas.pack()
         self.frame=Frame(raiz)
         self.frame.pack()
         self.altura = 100 # Altura do canvas        # Desenho do relógio-----------------------------
         pol=self.canvas.create_polygon
         ret=self.canvas.create_rectangle
         self.texto=self.canvas.create_text
         self.fonte=('BankGothic Md BT','20','bold')
         pol(10, self.altura-10,
             40, self.altura-90,
             160, self.altura-90,
             190, self.altura-10, fill='darkblue')
         pol(18, self.altura-15,
             45, self.altura-85, 

155, self.altura-85,
             182, self.altura-15, fill='dodgerblue')
         ret(50, self.altura-35,
             90, self.altura-65, fill='darkblue', outline='')
         ret(115, self.altura-35,
             150, self.altura-65, fill='darkblue', outline='')
         self.texto(100, self.altura-50, text=':',
               font=self.fonte, fill='yellow')
         # Fim do desenho do relógio-----------------------
         self.mostrar=Button(self.frame, text='Que horas são?',
           command=self.mostra,
           font=('Comic Sans MS', '11', 'bold'),
           fg='darkblue', bg='deepskyblue')
         self.mostrar.pack(side=LEFT) 
    def mostra(self):
         self.canvas.delete('digitos_HORA')
         self.canvas.delete('digitos_MIN')
         HORA = str( localtime()[3] )
         MINUTO = str( localtime()[4] )
         self.texto(67.5, self.altura-50, text=HORA, fill='yellow',
                font=self.fonte, tag='digitos_HORA')
         self.texto(132.5, self.altura-50, text=MINUTO, fill='yellow',
                font=self.fonte, tag='digitos_MIN') 

instancia=Tk() 
Horas(instancia) 
instancia.mainloop()

class Horas: 
    def __init__(self,raiz):
         self.canvas=Canvas(raiz, width=200, height=100)
         self.canvas.pack()
         self.frame=Frame(raiz)
         self.frame.pack()
         self.altura = 100 # Altura do canvas        # Desenho do relógio-----------------------------
         pol=self.canvas.create_polygon
         ret=self.canvas.create_rectangle
         self.texto=self.canvas.create_text
         self.fonte=('BankGothic Md BT','20','bold')
         pol(10, self.altura-10,
             40, self.altura-90,
             160, self.altura-90,
             190, self.altura-10, fill='darkblue')
         pol(18, self.altura-15,
             45, self.altura-85,
155, self.altura-85,
             182, self.altura-15, fill='dodgerblue')
         ret(45, self.altura-35,
             90, self.altura-70, fill='darkblue', outline='')
         ret(110, self.altura-35,
             155, self.altura-70, fill='darkblue', outline='')
         self.texto(100, self.altura-50, text=':',
               font=self.fonte, fill='yellow')
         # Fim do desenho do relógio-----------------------
         self.mostrar=Button(self.frame, text='Que horas são?',
         command=self.mostra,
         font=('Comic Sans MS', '11', 'bold'),
         fg='darkblue', bg='deepskyblue')
         self.mostrar.pack(side=LEFT) 
    def mostra(self):
         self.canvas.delete('digitos_HORA')
         self.canvas.delete('digitos_MIN')
         HORA = str( localtime()[3] )
         MINUTO = str( localtime()[4] )
         self.texto(67.5, self.altura-50, text=HORA, fill='yellow',
                font=self.fonte, tag='digitos_HORA')
         self.texto(132.5, self.altura-50, text=MINUTO, fill='yellow',
                font=self.fonte, tag='digitos_MIN') 
instancia=Tk() 
Horas(instancia) 
instancia.mainloop() 
         

class Pacman: 
    def __init__(self, raiz):
            self.canvas=Canvas(raiz, height=200, width=200,
         
             takefocus=1, bg='deepskyblue',
         
             highlightthickness=0)
            self.canvas.bind('<Left>', self.esquerda)
            self.canvas.bind('<Right>', self.direita)
            self.canvas.bind('<Up>', self.cima)
            self.canvas.bind('<Down>', self.baixo)
            self.canvas.focus_force()
            self.canvas.pack()
            # Desenho da carinha----------------------------------
            self.canvas.create_oval(80, 80, 120, 130,
         
         
         tag='bola', fill='yellow')
            self.canvas.create_oval(93, 100, 98, 95,
         
         
            tag='bola', fill='blue')

            self.canvas.create_oval(102, 100, 107, 95,
         
            tag='bola', fill='blue')
            self.canvas.create_arc(92, 87, 108, 107, tag='bola',
         
           start=220, extent=100, style=ARC)
    def esquerda(self, event): self.canvas.move('bola', -10, 0) 
    def direita(self, event): self.canvas.move('bola', 10, 0) 
    def cima(self, event): self.canvas.move('bola', 0, -10) 
    def baixo(self, event): self.canvas.move('bola', 0, 10) 
instancia=Tk() 
Pacman(instancia) 
instancia.mainloop()


class Nao_Redimensiona: 
    def __init__(self,janela):
            janela.resizable(width=False, height=True)
            janela.title('Não redimensiona!')
            Canvas(janela, width=200, height=100, bg='moccasin').pack() 
class Tamanhos_Limite: 
    def __init__(self,janela):
            janela.maxsize(width=300, height=300)
            janela.minsize(width=50, height=50)
            janela.title('Tamanhos limitados!')
            Canvas(janela, width=200, height=100, bg='moccasin').pack() 
            
inst1, inst2 = Tk(), Tk() 
Nao_Redimensiona(inst1) 
Tamanhos_Limite(inst2) 
inst1.mainloop()
inst2.mainloop()