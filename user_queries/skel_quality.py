from tkinter import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from neuprint import Client, skeleton
from neuprint import fetch_synapses, NeuronCriteria as NC
import pandas as pd
from os.path import isfile

class MyWindow:
    # get list of neurons
    home_dir = '/Users/gs697/Research/mitochondriaMorphologyPosition' 
    token_id = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6ImdhcnJldHQuc2FnZXJAeWFsZS5lZHUiLCJsZXZlbCI6Im5vYXV0aCIsImltYWdlLXVybCI6Imh0dHBzOi8vbGgzLmdvb2dsZXVzZXJjb250ZW50LmNvbS9hLS9BT2gxNEdpTGNqZXlHYWNnS3NPcTgzdDNfczBoTU5sQUtlTkljRzdxMkU5Rz1zOTYtYz9zej01MD9zej01MCIsImV4cCI6MTgwMTAxNzUwNn0.dzq7Iy01JwSWbKq-Qvi8ov7Hwr0-ozpYeSnOsUD-Mx0"
    c = Client('neuprint.janelia.org', dataset='hemibrain:v1.2.1', token=token_id)

    def __init__(self, window):
        window.geometry("1300x980+300+10") # "widthxheight+XPOS+YPOS"

        lc_nums = np.array( [4,6] )
        lc_nums = np.concatenate( (lc_nums, np.arange(9,46+1,1)) )
        results = np.array( [] )
        for lc_num in lc_nums:
            q = f"""\
                MATCH (a:Neuron)
                WHERE a.status = "Traced" AND a.cropped = False AND a.type = 'LC{lc_num}'
                RETURN a.bodyId AS bodyID, a.type AS neuron_type
            """
            this_results = MyWindow.c.fetch_custom(q).to_numpy()
            if results.shape[0] == 0:
                results = this_results
            else:
                results = np.concatenate( (results, this_results), axis=0 )
        self.file_name = MyWindow.home_dir + "/saved_data/neuron_quality.csv"

        if isfile(self.file_name):
            self.param_df = pd.read_csv(self.file_name)
            already_analyzed = np.isin( results[:,0], self.param_df['bodyId'].to_numpy() )
            results = results[ ~already_analyzed ]
        else:
            self.param_df = pd.DataFrame( data=[], columns = ['bodyId', 'neuron_type', 'has_soma', 'has_axon', 'has_dendrite'] )
        self.results = results

        # plot skeleton on the left
        self.fig, self.axes = plt.subplots(nrows=2, ncols=2, figsize=(15,15) )
        self.plth = FigureCanvasTkAgg(self.fig, window)
        self.plth.get_tk_widget().pack(side=LEFT, fill=Y)
        self.fig.delaxes(self.axes[1][1])
        MyWindow.plt_skel(self, window)

        # display buttons for user
        init_x = 850
        init_y = 520
        right_btn_space = 100
        vert_spacing = 35

        # ask user if neuron has soma
        self.soma_lbl = Label(window, text="Has Soma?", fg='black', font=("Helvetica", 16))
        self.soma_lbl.place(x=init_x+30,y=init_y)

        self.soma_var = IntVar()
        self.soma_no_btn=Radiobutton(window, text="NO", variable=self.soma_var, value=1)
        self.soma_no_btn.place(x=init_x+right_btn_space,y=init_y+vert_spacing)

        self.soma_yes_btn=Radiobutton(window, text="YES", variable=self.soma_var, value=2)
        self.soma_yes_btn.place(x=init_x,y=init_y+vert_spacing)

        # ask user if axon and/or dendrite arbors are cropped
        dy = 110
        vert_btn_spacing = 30
        self.arbor_lbl = Label(window, text="Has Arbors?", fg='black', font=("Helvetica", 16))
        self.arbor_lbl.place(x=init_x+30,y=init_y+dy)

        self.arbor_var = IntVar()

        self.both_arbor_btn = Radiobutton(window, text="Both", variable=self.arbor_var, value=1)
        self.both_arbor_btn.place(x=init_x,y=init_y+dy+vert_spacing)

        self.no_arbor_btn = Radiobutton(window, text="Neither/IDK", variable=self.arbor_var, value=2)
        self.no_arbor_btn.place(x=init_x+right_btn_space,y=init_y+dy+vert_spacing)

        self.LO_arbor_btn = Radiobutton(window, text="Only LO", variable=self.arbor_var, value=3)
        self.LO_arbor_btn.place(x=init_x,y=init_y+dy+vert_spacing+vert_btn_spacing)

        self.CB_arbor_btn = Radiobutton(window, text="Only CB", variable=self.arbor_var, value=4)
        self.CB_arbor_btn.place(x=init_x+right_btn_space,y=init_y+dy+vert_spacing+vert_btn_spacing)

        # submit answer button
        # lambda: self.plot(canvas,ax)
        self.submit_btn=Button(window, text="Submit Response", fg='blue', width=30, height=3, font=("Helvetica", 16), command= lambda: self.submit_neuron(window))
        self.submit_btn.place(x=init_x-100,y=init_y+dy+vert_spacing+vert_btn_spacing*2+40)


    def plt_skel(self, window):
        bodyId, neuron_type = self.results[0]
        title_name = "Number Remaining Neurons : " + str(self.results.shape[0]) + ";  bodyId = " + str(bodyId) + ";  type = " + str(neuron_type)
        window.title(title_name)

        # plot 3 different projections of the neuron
        s_pandas = MyWindow.c.fetch_skeleton( bodyId, format='pandas', heal=True, with_distances=False)
        skeleton.reorient_skeleton( s_pandas,  use_max_radius=True )
        synapse_sites = fetch_synapses( NC(bodyId=bodyId) )
        synapse_coords = np.array( [synapse_sites['x'],synapse_sites['y'],synapse_sites['z']] )
        synapse_loc = synapse_sites['roi']


        s_np = s_pandas.to_numpy()
        root_idx = np.where(s_np[:,5]==-1)[0][0]

        idxs = np.arange(s_np.shape[0])
        idxs = idxs[ idxs != root_idx]
        #idxs = idxs[ np.arange(0,len(idxs),2) ] # plot every other neuron segment
        down_idxs = np.zeros( (len(idxs),), dtype=int )
        for i, idx in enumerate(idxs):
            down_idxs[i] = np.where( s_np[idx,5] == s_np[:,0] )[0][0]

        fontsize = 15
        synapse_site_size = 10
        dim_pairs = [[1,2],[1,3],[2,3]]
        for i, dim_pair in enumerate(dim_pairs):
            row = i % 2
            col = 1 if i==2 else 0
            self.axes[row,col].clear()
            self.axes[row,col].scatter( s_np[root_idx,dim_pair[0]], s_np[root_idx,dim_pair[1]], s=50, c='r' )

            dim_1 = s_np[idxs, dim_pair[0]], s_np[down_idxs, dim_pair[0]]
            dim_2 = s_np[idxs, dim_pair[1]], s_np[down_idxs, dim_pair[1]]
            self.axes[row,col].plot( dim_1, dim_2, 'k', linewidth= 0.5 )

            lob_synapses = np.array(synapse_sites['roi'] == 'LO(R)')
            if np.any(lob_synapses):
                lob_synapse_idxs = np.where(lob_synapses)[0]
                if np.sum(lob_synapses) > 30:
                    # only plot every 4th synapse
                    lob_synapse_idxs = lob_synapse_idxs[ np.arange(0,len(lob_synapse_idxs),4) ]
                self.axes[row,col].scatter( synapse_coords[dim_pair[0]-1, lob_synapse_idxs], synapse_coords[dim_pair[1]-1, lob_synapse_idxs], s=2, c='b', label='LO' )
            if np.any(~lob_synapses):
                nonlob_synapse_idxs = np.where(~lob_synapses)[0]
                if np.sum(~lob_synapses) > 30:
                    # only plot every 4th synapse
                    nonlob_synapse_idxs = nonlob_synapse_idxs[ np.arange(0,len(nonlob_synapse_idxs),4) ]
                self.axes[row,col].scatter( synapse_coords[dim_pair[0]-1,nonlob_synapse_idxs], synapse_coords[dim_pair[1]-1,nonlob_synapse_idxs], s=2, c='g', label='CB' )
            lgnd = self.axes[row,col].legend(fontsize=fontsize)
            self.axes[row,col].set_xticks([]); self.axes[row,col].set_yticks([])

            #change the marker size manually for both lines
            lgnd.legendHandles[0]._sizes = [30]
            if np.any(lob_synapses) and np.any(~lob_synapses):
                lgnd.legendHandles[1]._sizes = [30]
    def submit_neuron(self, window):
        arbor_val = self.arbor_var.get()
        soma_val = self.soma_var.get()
        if (soma_val>0) and (arbor_val>0):
            bodyId, neuron_type = self.results[0]
            has_soma = soma_val == 2
            has_axon = np.any(arbor_val == np.array([1, 4]) )
            has_dendrite = np.any(arbor_val == np.array([1, 3]) )
            self.param_df.loc[ self.param_df.shape[0] ] = [bodyId, neuron_type, has_soma, has_axon, has_dendrite]
            self.param_df.to_csv(self.file_name, index=False)

            if self.results.shape[0] == 1:
                # that was the last neuron, so save dataframe and close window
                window.destroy()
                assert False, 'Congrats, you finshed'
            else:
                # ensures results has at least one row
                self.results = self.results[1:].reshape((self.results.shape[0]-1,2))

            MyWindow.plt_skel(self, window)
            self.plth.draw()

        self.soma_var.set(0)
        self.arbor_var.set(0)


window = Tk()
win = MyWindow(window)
window.mainloop()
