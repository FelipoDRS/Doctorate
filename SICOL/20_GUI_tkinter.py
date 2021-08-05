# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 16:40:21 2021

@author: Felipo Soares
"""
import tkinter as tk
from pickle import load
from sklearn.ensemble import GradientBoostingRegressor
import sklearn.neighbors._typedefs
import sklearn.neighbors._quad_tree
import sklearn.tree
import sklearn.tree._utils
path_model=r'E:\coppe\SICOL_novo\modelos\20210625'

class SICOL_GUI:
    def __init__(self,janela):
        janela.minsize(width=450, height=150)
        janela.title('SICOL')
        janela.configure(bg='dark grey')

        self.frame1=tk.Frame(janela)         
        self.frame1.grid(row=1,column=1)         
        self.frame2=tk.Frame(janela)         
        self.frame2.grid(row=2,column=1)        
        self.frame3=tk.Frame(janela)         
        self.frame3.grid(row=3,column=1)         
        fonte_campo=('Verdana','8','bold')
        fonte_titulo=('Verdana','12','bold')
        tk.Label(self.frame1,text='Densidade da carga (g/cm^3)', fg='darkblue',bg='dark grey',font=fonte_campo, height=3).pack()
        self.d_C=tk.Entry(self.frame1,width=10,font=fonte_campo)
        self.d_C.insert(0,0.8869)
        self.d_C.focus_force() # Para o foco começar neste campo        
        self.d_C.pack(side=tk.BOTTOM)
        self.select_D_R=tk.IntVar()
        
        tk.Radiobutton(self.frame2, text = 'Densidade do rafinado predita', variable = self.select_D_R,
                value = 0, indicator = 0,
                background = "light blue").pack()
        tk.Radiobutton(self.frame2, text = 'Densidade do rafinado especificada', variable = self.select_D_R,
                value = 1, indicator = 0,
                background = "light blue").pack()

        tk.Label(self.frame3,text='Densidade do rafinado (g/cm^3)',bg='dark grey', fg='darkblue',font=fonte_campo, height=3).pack()
        self.d_R=tk.Entry(self.frame3,width=10,font=fonte_campo)
        self.d_R.insert(0,0.8473)
        self.d_R.pack(side=tk.BOTTOM)
        self.frame4=tk.Frame(janela)         
        self.frame4.grid(row=1,column=2) 
        tk.Label(self.frame4,text='Desaromatização',bg='dark grey', fg='darkblue',font=fonte_titulo).pack()
        self.frame5=tk.Frame(janela)         
        self.frame5.grid(row=2,column=2) 
        self.frame6=tk.Frame(janela)         
        self.frame6.grid(row=3,column=2) 
        tk.Label(self.frame5,text='Temperatura',bg='dark grey', fg='darkblue',font=fonte_campo).pack(side=tk.LEFT)
        self.temp_da=tk.Entry(self.frame5,width=10,font=fonte_campo)
        self.temp_da.insert(0,79.56)
        self.temp_da.pack(side=tk.RIGHT)

        tk.Label(self.frame6,text='Razão solvente óleo',bg='dark grey', fg='darkblue',font=fonte_campo).pack(side=tk.LEFT)
        self.RSO_da=tk.Entry(self.frame6,width=10,font=fonte_campo)
        self.RSO_da.insert(0,7)
        self.RSO_da.pack(side=tk.RIGHT)
        
        self.frame7=tk.Frame(janela)         
        self.frame7.grid(row=1,column=3) 
        tk.Label(self.frame7,text='Desparafinacao',bg='dark grey', fg='darkblue',font=fonte_titulo).pack()
        self.frame8=tk.Frame(janela)         
        self.frame8.grid(row=2,column=3) 
        self.frame9=tk.Frame(janela)         
        self.frame9.grid(row=3,column=3) 
        tk.Label(self.frame8,text='Temperatura',bg='dark grey', fg='darkblue',font=fonte_campo).pack(side=tk.LEFT)
        self.temp_dp=tk.Entry(self.frame8,width=10,font=fonte_campo)
        self.temp_dp.insert(0,-13.29)
        self.temp_dp.pack(side=tk.RIGHT)
        self.frame18=tk.Frame(janela)         
        self.frame18.grid(row=4,column=3) 
        tk.Label(self.frame18,text='Razão de Lavagem',bg='dark grey', fg='darkblue',font=fonte_campo).pack(side=tk.LEFT)
        self.Lav_dp=tk.Entry(self.frame18,width=10,font=fonte_campo)
        self.Lav_dp.insert(0,2)
        self.Lav_dp.pack(side=tk.RIGHT)

        tk.Label(self.frame9,text='Razão solvente óleo',bg='dark grey', fg='darkblue',font=fonte_campo).pack(side=tk.LEFT)
        self.RSO_dp=tk.Entry(self.frame9,width=10,font=fonte_campo)
        self.RSO_dp.insert(0,4)
        self.RSO_dp.pack()
        
        self.frame10=tk.Frame(janela)         
        self.frame10.grid(row=5,column=2,sticky=tk.W) 
        self.frame11=tk.Frame(janela)         
        self.frame11.grid(row=6,column=2,sticky=tk.W) 
        self.frame12=tk.Frame(janela)         
        self.frame12.grid(row=7,column=2,sticky=tk.W) 
        self.frame13=tk.Frame(janela)         
        self.frame13.grid(row=8,column=2,sticky=tk.W)
        
        self.d_R1=tk.Label(self.frame10,bg='dark grey',text='Densidade do rafinado (g/cm^3): ', font=fonte_campo)
        self.d_R1.pack()
        self.IR_R=tk.Label(self.frame11,bg='dark grey',text='Índice de refração: ', font=fonte_campo)
        self.IR_R.pack()
        self.IV_R=tk.Label(self.frame12,bg='dark grey',text='Índice de viscosidade:', font=fonte_campo)
        self.IV_R.pack()
        self.RendR=tk.Label(self.frame13,bg='dark grey',text='Rendimento do rafinado: ', font=fonte_campo)
        self.RendR.pack()
                
        self.frame14=tk.Frame(janela)         
        self.frame14.grid(row=5,column=3,sticky=tk.W) 
        self.frame15=tk.Frame(janela)         
        self.frame15.grid(row=6,column=3,sticky=tk.W) 
        self.frame16=tk.Frame(janela)         
        self.frame16.grid(row=7,column=3,sticky=tk.W) 
        self.frame17=tk.Frame(janela)         
        self.frame17.grid(row=8,column=3,sticky=tk.W)
        
        self.d_DP1=tk.Label(self.frame14,bg='dark grey',text='Densidade do desparafinado (g/cm^3): ', font=fonte_campo)
        self.d_DP1.pack(side=tk.LEFT)
        self.IR_DP=tk.Label(self.frame15,bg='dark grey',text='Índice de refração: ', font=fonte_campo)
        self.IR_DP.pack(side=tk.LEFT)
        self.IV_DP=tk.Label(self.frame16,bg='dark grey',text='Índice de viscosidade:', font=fonte_campo)
        self.IV_DP.pack(side=tk.LEFT)
        self.RendDP=tk.Label(self.frame17,bg='dark grey',text='Rendimento do desparafinado: ', font=fonte_campo)
        self.RendDP.pack(side=tk.LEFT)
        
        self.frame19=tk.Frame(janela)         
        self.frame19.grid(row=2,column=5,sticky=tk.W)
        self.calc=tk.Button(self.frame19, width=18, command=self.calcular, text='Calcular').pack()
        self.mod_d_R=load(open(path_model+'\\GBR20210625_d_R.joblib','rb'))
        self.mod_IR_R=load(open(path_model+'\\GBR20210625_IR_R.joblib','rb'))
        self.mod_IV_R=load(open(path_model+'\\GBR20210625_IV_R.joblib','rb'))
        self.mod_RendR=load(open(path_model+'\\GBR20210625_RendR.joblib','rb'))
        self.mod_d_DP=load(open(path_model+'\\GBR20210625_d_DP.joblib','rb'))
        self.mod_IR_DP=load(open(path_model+'\\GBR20210625_IR_DP.joblib','rb'))
        self.mod_IV_DP=load(open(path_model+'\\GBR20210625_IV_DP.joblib','rb'))
        self.mod_RendDP=load(open(path_model+'\\GBR20210625_RendDP.joblib','rb'))
        
    def calcular(self) :
        X_da=[[self.temp_da.get(),self.RSO_da.get(), self.d_C.get()]]
        yhat_da=[self.mod_d_R.predict(X_da),
              self.mod_IV_R.predict(X_da),
              self.mod_IR_R.predict(X_da),
              self.mod_RendR.predict(X_da)]
        if self.select_D_R.get()==1:
            
            X_dp=[[self.d_R.get(), self.temp_da.get(),self.RSO_da.get(), 
              self.temp_dp.get(),self.RSO_dp.get(),self.Lav_dp.get()]]

        else:
            X_dp=[[yhat_da[0], self.temp_da.get(),self.RSO_da.get(), 
              self.temp_dp.get(),self.RSO_dp.get(),self.Lav_dp.get()]]
        
        yhat_dp=[self.mod_d_DP.predict(X_dp),
              self.mod_IV_DP.predict(X_dp),
              self.mod_IR_DP.predict(X_dp),
              self.mod_RendDP.predict(X_dp)]
              
        self.IR_R['text']='Índice de refração: '+str(yhat_da[2])[1:6]
        self.d_R1['text']='Densidade do rafinado (g/cm^3): '+str(yhat_da[0])[1:6]
        self.IV_R['text']='Índice de viscosidade:'+str(yhat_da[1])[1:4]
        self.RendR['text']='Rendimento do rafinado: '+str(yhat_da[3])[1:5]
        
        self.d_DP1['text']='Densidade do desparafinado (g/cm^3): '+str(yhat_dp[0])[1:5]
        self.IR_DP['text']='Índice de refração: '+str(yhat_dp[2])[1:5]
        self.IV_DP['text']='Índice de viscosidade:'+str(yhat_dp[1])[1:4]
        self.RendDP['text']='Rendimento do desparafinado: '+str(yhat_dp[3])[1:5]
        
        
                
raiz=tk.Tk() 
SICOL_GUI(raiz) 
raiz.mainloop()
