import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askopenfilename, askopenfilenames, askdirectory
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import peaks as pk_elab

class OpenS:
    """Class for graphical open file dialog

    Attributes
    ----------
    self.filename : str
        filename (filepath)

    Methods
    -------
    None
    """
    def __init__(self, parent, ftype='all'):
        """OpenS class constructor

        Parameters
        ----------
        parent : tk.Tk (or None)
            root window
        ftype : str
            select specific filetypes to filter [choices: 'csv', 'n42', 'psp'] (default 'all')

        Return
        ------
        None
        """
        _types = (('CSV files','*.csv'), ('N42 formatted spectra files','*.n42'), ('Preserved spectra objects','*.psp'), ('Preserved elaborated spectra','*.pel'))
        if ftype != 'all':
            filetypes = tuple([item for item in _types if item[1].endswith(ftype)])
        else:
            filetypes = ()
        self.filename = askopenfilename(title=f'Recall single file, type ({ftype})', filetypes=filetypes)

class OpenM:
    """Class for graphical open files dialog

    Attributes
    ----------
    self.filenames : tuple
        tuple of filenames (filepaths)

    Methods
    -------
    None
    """
    def __init__(self, parent, ftype='all'):
        """OpenM class constructor

        Parameters
        ----------
        parent : tk.Tk (or None)
            root window
        ftype : str
            select specific filetypes to filter [choices: 'csv', 'n42', 'psp'] (default 'all')

        Return
        ------
        None
        """
        _types = (('CSV files','*.csv'), ('N42 formatted spectra files','*.n42'), ('Preserved spectra objects','*.psp'), ('Preserved elaborated spectra','*.pel'))
        if ftype != 'all':
            filetypes = tuple([item for item in _types if item[1].endswith(ftype)])
        else:
            filetypes = ()
        self.filenames = askopenfilenames(parent=parent, title=f'Recall multiple files, type ({ftype})', filetypes=filetypes)


class OpenD:
    """Class for graphical open files dialog

    Attributes
    ----------
    self.filenames : tuple
        tuple of filenames (filepaths)

    Methods
    -------
    None
    """
    def __init__(self, parent):
        """OpenM class constructor

        Parameters
        ----------
        parent : tk.Tk (or None)
            root window
        ftype : str
            select specific filetypes to filter (default 'all')

        Return
        ------
        None
        """
        self.folder = askdirectory(parent=parent, title='Select save folder')

class RegionFinder:
    """Class to graphically identify regions of interest in the spectrum and fit peaks

    Attributes
    ----------
    self.spc_elab : None
        descr
    self.spectrum : spectra.GammaSpectrum
        spectrum to be elaborated
    self.bkg : spectra.GammaSpectrum (or None)
        spectrum used for background subtraction
    self.regions : list
        list of regions identified with the spectrum counts
    self.iso_db : pandas.DataFrame
        dataframe with relevant isotope information

    Methods
    -------
    _update_choice()
        descr
    _end_everithing()
        descr
    on_scale_changed()
        descr
    on_motion()
        descr
    _add_region_to_list()
        descr
    _remove_region()
        descr
    update_view()
        descr
    _save_modify_info()
        descr
    _show_fit_preview()
        descr
    """
    def __init__(self, M, spc, bkg=None):
        M.title('Region finder')
        self.spc_elab = None

        self.spectrum = spc
        self.bkg = bkg
        self.regions = []

        self.iso_db = pk_elab.load_isotope_database(short=True)

        main_header = tk.Frame(M)
        tk.Label(main_header, text='spectrum filename', anchor=tk.W).grid(row=0, column=0, sticky=tk.W)
        EEE = tk.Entry(main_header, width=130, justify='right')
        EEE.delete(0, tk.END)
        EEE.insert(0, f'{self.spectrum.o_filename}')
        EEE.grid(row=1, column=0, sticky=tk.EW)
        EEE['state'] = 'readonly'

        main_header.pack(anchor=tk.NW, fill=tk.X, expand=True, padx=5)

        add_info = tk.Frame(M)
        tk.Label(add_info, text='real time / s', anchor=tk.W, width=12).grid(row=0, column=0, sticky=tk.W)
        tk.Label(add_info, text='live time / s', anchor=tk.W, width=12).grid(row=1, column=0, sticky=tk.W)

        tk.Label(add_info, text=f'{self.spectrum.real_time/1E12:.1f}', anchor=tk.E, width=10).grid(row=0, column=1, sticky=tk.E)
        tk.Label(add_info, text=f'{self.spectrum.live_time/1E12:.1f}', anchor=tk.E, width=10).grid(row=1, column=1, sticky=tk.E)

        tk.Frame(add_info).grid(row=0, column=2, padx=20)

        tk.Label(add_info, text='ext_deadtime / μs', anchor=tk.W, width=14).grid(row=0, column=3, sticky=tk.W)
        tk.Label(add_info, text='channels / 1', anchor=tk.W, width=14).grid(row=1, column=3, sticky=tk.W)

        tk.Label(add_info, text=self.spectrum.extended_dead_time, anchor=tk.E, width=12).grid(row=0, column=4, sticky=tk.E)
        tk.Label(add_info, text=f'{self.spectrum.channels}', anchor=tk.E, width=12).grid(row=1, column=4, sticky=tk.E)

        tk.Frame(add_info).grid(row=0, column=5, padx=20)

        tk.Label(add_info, text='total counts / 1', anchor=tk.W, width=12).grid(row=0, column=6, sticky=tk.W)
        tk.Label(add_info, text='count rate / s-1', anchor=tk.W, width=12).grid(row=1, column=6, sticky=tk.W)
        tk.Label(add_info, text=f'{self.spectrum.defined_spectrum_integral()}', anchor=tk.E, width=16).grid(row=0, column=7, sticky=tk.E)
        ICR, uICR = self.spectrum.input_count_rate()
        tk.Label(add_info, text=f'{ICR:.1f} ({uICR:.1f})', anchor=tk.E, width=16).grid(row=1, column=7, sticky=tk.E)

        add_info.pack(anchor=tk.NW, fill=tk.X, padx=5)

        bkg_name = tk.Frame(M)

        tk.Label(bkg_name, text='spectrum background', anchor=tk.W).grid(row=0, column=0, sticky=tk.W)
        EEEBKG = tk.Entry(bkg_name, width=130, justify='right')
        EEEBKG.delete(0, tk.END)
        if self.bkg is not None:
            f_name_BKG = f'{self.bkg.o_filename}'
        else:
            f_name_BKG = '-'
        EEEBKG.insert(0, f_name_BKG)
        EEEBKG.grid(row=1, column=0, sticky=tk.EW)
        EEEBKG['state'] = 'readonly'

        bkg_name.pack(anchor=tk.NW, fill=tk.X, expand=True, padx=5, pady=3)

        graph_part = tk.Frame(M)

        self.figure = Figure(figsize=(8, 4))
        self.figure.patch.set_alpha(0.0)
        self.ax = self.figure.add_subplot(111)
        Figur = tk.Frame(graph_part)
        Figur.grid(row=0, column=0, sticky=tk.NSEW)
        self.canvas = FigureCanvasTkAgg(self.figure, master=Figur)
        self.ax.set_xlabel('channel / 1')
        self.figure.tight_layout()
        self.canvas.draw()
        self.canvas.get_tk_widget().configure(background=graph_part.cget('bg'))
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.ax.plot(self.spectrum.counts)
        fbx = np.array([np.nan, np.nan])
        self.ax.set_xlim(0, len(self.spectrum.counts)-1)
        self._ceiling = np.max(self.spectrum.counts)*1.08 + 10
        self.ax.set_ylim(0, self._ceiling)
        self.int_area = self.ax.fill_between(fbx, self._ceiling, y2=0, alpha=0.3, color='green')

        self.figure.tight_layout()
        self.canvas.draw()

        graph_part.pack(anchor=tk.NW, fill=tk.BOTH, expand=True)

        self.navigation_scale = tk.Scale(M, from_=-1, to=len(self.spectrum.counts)-1, width=10, length=500, orient=tk.HORIZONTAL, resolution=1)
        self.navigation_scale.pack(anchor=tk.NW, fill=tk.X, expand=True)
        self.navigation_scale.set(-1)

        self.coord = tk.Label(M, text='CHANNEL= 0 | COUNTS= 0 (0 keV)', anchor=tk.E)
        self.coord.pack(anchor=tk.NE, fill=tk.X, expand=True)

        elab_stuff = tk.LabelFrame(M, labelwidget=tk.Label(M, text='regions'), relief='solid', bd=2, padx=4, pady=4)

        button_stuff = tk.Frame(elab_stuff)

        self.reg_CB = ttk.Combobox(button_stuff, state='readonly', width=40)
        self.reg_CB.pack(side=tk.LEFT, padx=5)
        self.reg_CB['values'] = [item.__repr__() for item in self.regions]

        self.howmany = tk.Label(button_stuff, text=f'{len(self.regions)}', width=4)
        self.howmany.pack(side=tk.LEFT, padx=3)

        add_button = tk.Button(button_stuff, text='ADD', width=10)
        add_button.pack(side=tk.LEFT)

        remove_button = tk.Button(button_stuff, text='REMOVE', width=10)
        remove_button.pack(side=tk.LEFT)

        button_stuff.grid(row=0, column=0, sticky=tk.W)

        single_region_info = tk.Frame(elab_stuff)
        tk.Label(single_region_info, text='start region / 1', width=14).grid(row=0, column=0)
        tk.Label(single_region_info, text='end region / 1', width=14).grid(row=0, column=1)
        tk.Label(single_region_info, text='centroid / 1', width=14).grid(row=0, column=2)
        tk.Label(single_region_info, text='energy / keV', width=14).grid(row=0, column=3)
        tk.Label(single_region_info, text='activity / Bq', width=14).grid(row=0, column=4)
        tk.Label(single_region_info, text='activity_unc / Bq', width=14).grid(row=0, column=5)
        tk.Label(single_region_info, text='isotope label', width=14).grid(row=0, column=6)
        tk.Label(single_region_info, text='function', width=14).grid(row=0, column=7)

        self.Start_Reg_E = tk.Spinbox(single_region_info, width=12, from_=0, to=len(self.spectrum.counts)-2, increment=1)
        self.Start_Reg_E.grid(row=1, column=0, padx=3)
        self.End_Reg_E = tk.Spinbox(single_region_info, width=12, from_=1, to=len(self.spectrum.counts)-1, increment=1)
        self.End_Reg_E.grid(row=1, column=1, padx=3)
        self.Centr_Reg_E = tk.Spinbox(single_region_info, width=12, from_=1, to=len(self.spectrum.counts)-2, increment=1)
        self.Centr_Reg_E.grid(row=1, column=2, padx=3)

        self.Energy_Reg_E = ttk.Combobox(single_region_info, width=12, state='readonly')
        self.Energy_Reg_E['values'] = []
        self.Energy_Reg_E.set('')
        self.Energy_Reg_E.grid(row=1, column=3, padx=3)

        self.Act_Reg_E = tk.Spinbox(single_region_info, width=12, from_=1, to=500000, increment=1)
        self.Act_Reg_E.grid(row=1, column=4, padx=3)
        self.Act_unc_Reg_E = tk.Spinbox(single_region_info, width=12, from_=1, to=100000, increment=1)
        self.Act_unc_Reg_E.grid(row=1, column=5, padx=3)

        self.ITP_Reg_E = ttk.Combobox(single_region_info, width=16, state='readonly')
        self.ITP_Reg_E['values'] = sorted(self.iso_db['isotope'].unique())
        self.ITP_Reg_E.set('')
        self.ITP_Reg_E.grid(row=1, column=6, padx=3)

        self._functions_CCB = ttk.Combobox(single_region_info, width=18, state='readonly')
        self._functions_CCB['values'] = ('normal_step', 'normal_slope', 'normal_stepcurve', 'normal_slopecurve')
        self._functions_CCB.grid(row=1, column=7, padx=3)
        self._functions_CCB.set(self._functions_CCB['values'][0])

        svm_button = tk.Button(single_region_info, text='SAVE/MODIFY', width=12)
        svm_button.grid(row=1, column=8, padx=3)

        fitpw_button = tk.Button(single_region_info, text='FIT PREVIEW', width=12)
        fitpw_button.grid(row=2, column=8, padx=3)

        single_region_info.grid(row=1, column=0, columnspan=5, sticky=tk.W)

        elab_stuff.pack(anchor=tk.NW, fill=tk.X, expand=True, padx=5, pady=5)

        Confirm_ALL_BT = tk.Button(M, text='CONFIRM REGIONS AND CLOSE', width=30)
        Confirm_ALL_BT.pack(pady=10, padx=5)

        main_header.columnconfigure(0, weight=1)
        bkg_name.columnconfigure(0, weight=1)
        graph_part.columnconfigure(0, weight=1)
        graph_part.rowconfigure(0, weight=1)

        sid = self.canvas.mpl_connect('motion_notify_event', lambda event='motion_notify_event' : self.on_motion(event))
        self.navigation_scale.configure(command= lambda event : self.on_scale_changed())

        add_button.configure(command=lambda : self._add_region_to_list())
        remove_button.configure(command=lambda : self._remove_region())
        svm_button.configure(command=lambda : self._save_modify_info())
        fitpw_button.configure(command=lambda : self._show_fit_preview(M))
        self.reg_CB.bind('<<ComboboxSelected>>', lambda e='<<ComboboxSelected>>' : self.update_view())
        self.ITP_Reg_E.bind('<<ComboboxSelected>>', lambda e='<<ComboboxSelected>>' : self._update_choice())
        Confirm_ALL_BT.configure(command=lambda : self._end_everithing(M))

    def _update_choice(self):
        key = self.ITP_Reg_E.get()
        _filter = self.iso_db['isotope'] == key
        self.Energy_Reg_E['values'] = [f'{row:.2f}' for row in self.iso_db[_filter]['energy / keV']]
        self.Energy_Reg_E.set(self.Energy_Reg_E['values'][0])

    def _end_everithing(self, M):
        self.spc_elab = pk_elab.SpectrumElaboration(self.spectrum, self.regions, self.bkg)
        M.destroy()

    def on_scale_changed(self):
        center = self.navigation_scale.get()
        if center == -1:
            self.ax.set_xlim(0, self.spectrum.channels-1)
            self.ax.set_ylim(0, np.max(self.spectrum.counts)*1.08 + 10)
            self.figure.tight_layout()
            self.canvas.draw()
        else:
            hw = self.spectrum.channels // 200
            if hw < 50:
                hw = 50
            if center - hw <= 0:
                ll = 0
            else:
                ll = center - hw
            if center + hw >= self.spectrum.channels-1:
                hl = self.spectrum.channels-1
            else:
                hl = center + hw
            newx = np.arange(ll, hl, dtype=int)
            self.ax.set_xlim(ll, hl)
            self.ax.set_ylim(0, np.max(self.spectrum.counts[newx])*1.08 + 10)
            self.figure.tight_layout()
            self.canvas.draw()

    def on_motion(self, event):
        """Display x, y, coordinates while moving the mouse over the plot"""
        #mouse motion on spectrum profile
        text = ''
        if event.xdata is not None and event.ydata is not None:
            x = int(event.xdata)
            if x >= 0 and x < self.spectrum.channels:
                y = int(self.spectrum.counts[x])
                try:
                    W = np.array([x])
                    W = W[:,np.newaxis]**[_ for _ in range(len(self.spectrum.E_cal))]
                    Y = W @ self.spectrum.E_cal
                    Y = Y[0]
                except Exception:
                    Y = 0
                text = f'CHANNEL= {x} | COUNTS= {y} ({Y:.2f} keV)'
        self.coord.configure(text=text)

    def _add_region_to_list(self):
        self.reg_CB.set('')
        self.Energy_Reg_E['values'] = []
        self.Energy_Reg_E.set('')
        self.ITP_Reg_E.set('')

    def _remove_region(self):
        idx = self.reg_CB.current()

        try:
            self.regions.pop(idx)
        except IndexError:
            return

        self.regions.sort(key=lambda x : x.low)
        self.reg_CB['values'] = [item.__repr__() for item in self.regions]
        self.howmany.configure(text=f'{len(self.regions)}')

        ll, hl = 0, self.spectrum.channels - 1
        rl, rh = np.nan, np.nan

        if len(self.regions) == 0:
            self.reg_CB.set('')

        elif idx >= len(self.regions):
            reg = self.regions[0]
            self.reg_CB.set(self.reg_CB['values'][0])

            if reg.low - 10 < 0:
                ll = 0
            else:
                ll = reg.low - 10

            if reg.right + 10 > self.spectrum.channels - 1:
                hl = self.spectrum.channels -1
            else:
                hl = reg.right + 10
            rl, rh = reg.low, reg.right

        else:
            reg = self.regions[idx]
            self.reg_CB.set(self.reg_CB['values'][idx])

            if reg.low - 10 < 0:
                ll = 0
            else:
                ll = reg.low - 10

            if reg.right + 10 > self.spectrum.channels - 1:
                hl = self.spectrum.channels -1
            else:
                hl = reg.right + 10
            rl, rh = reg.low, reg.right

        newx = np.arange(ll, hl, dtype=int)
        self.ax.set_xlim(ll, hl)
        self.ax.set_ylim(0, np.max(self.spectrum.counts[newx])*1.08 + 10)
        self.int_area.set_data(np.array([rl, rh-1]), self._ceiling, 0)
        self.figure.tight_layout()
        self.canvas.draw()

    def update_view(self):
        reg = self.regions[self.reg_CB.current()]

        self.Start_Reg_E.delete(0, tk.END)
        self.Start_Reg_E.insert(0, int(reg.low))
        self.End_Reg_E.delete(0, tk.END)
        self.End_Reg_E.insert(0, int(reg.right))
        self.Centr_Reg_E.delete(0, tk.END)
        self.Centr_Reg_E.insert(0, int(reg.centroid))
        self.Energy_Reg_E.set(f'{reg.energy:.2f}')
        self.Act_Reg_E.delete(0, tk.END)
        self.Act_Reg_E.insert(0, int(reg.activity))
        self.Act_unc_Reg_E.delete(0, tk.END)
        self.Act_unc_Reg_E.insert(0, int(reg.activity_unc))
        self.ITP_Reg_E.set(reg.ilabel)
        try:
            indx = self._functions_CCB['values'].index(reg._funcs)
        except IndexError:
            indx = 0
        self._functions_CCB.set(self._functions_CCB['values'][indx])

        if reg.low - 10 < 0:
            ll = 0
        else:
            ll = reg.low - 10

        if reg.right + 10 > self.spectrum.channels - 1:
            hl = self.spectrum.channels -1
        else:
            hl = reg.right + 10

        newx = np.arange(ll, hl, dtype=int)
        self.ax.set_xlim(ll, hl)
        self.ax.set_ylim(0, np.max(self.spectrum.counts[newx])*1.08 + 10)
        self.int_area.set_data(np.array([reg.low, reg.right-1]), self._ceiling, 0)
        self.figure.tight_layout()
        self.canvas.draw()

    def _save_modify_info(self):

        try:
            low_limit = int(self.Start_Reg_E.get())
            high_limit = int(self.End_Reg_E.get())+1
            centroid = int(self.Centr_Reg_E.get())
            energy = float(self.Energy_Reg_E.get())
            activity = int(self.Act_Reg_E.get())
            activity_unc = int(self.Act_unc_Reg_E.get())
            ilabel = self.ITP_Reg_E.get()
            _fit_str = self._functions_CCB.get()
        except (TypeError, ValueError):
            return

        _RE = pk_elab.RegionElab(low_limit, high_limit, centroid, energy, activity, activity_unc, ilabel, _fit_str)

        if self.reg_CB.get() == '':
            self.regions.append(_RE)
        else:
            idx = self.reg_CB.current()
            self.regions[idx] = _RE
        #update
        self.regions.sort(key=lambda x : x.low)
        self.reg_CB['values'] = [item.__repr__() for item in self.regions]
        n_idx = self.regions.index(_RE)
        self.reg_CB.set(self.reg_CB['values'][n_idx])
        self.howmany.configure(text=f'{len(self.regions)}')

        if _RE.low - 10 < 0:
            ll = 0
        else:
            ll = _RE.low - 10

        if _RE.right + 10 > self.spectrum.channels - 1:
            hl = self.spectrum.channels -1
        else:
            hl = _RE.right + 10

        newx = np.arange(ll, hl, dtype=int)
        self.ax.set_xlim(ll, hl)
        self.ax.set_ylim(0, np.max(self.spectrum.counts[newx])*1.08 + 10)
        self.int_area.set_data(np.array([_RE.low, _RE.right-1]), self._ceiling, 0)
        self.figure.tight_layout()
        self.canvas.draw()

    def _show_fit_preview(self, M):
        idx = self.reg_CB.current()
        try:
            reg = self.regions[idx]
        except IndexError:
            print('no region is selected')
            return

        if self.bkg is None:
            BKG_counts = np.zeros(self.spectrum.channels)
        else:
            BKG_counts = np.array(self.bkg.counts/self.bkg.live_time*self.spectrum.live_time, dtype=int)

        X, Y, _ = reg.spectrum_slice(self.spectrum.counts, BKG_counts)
        try:
            Yvar = self.spectrum.var[X] + BKG_counts[X]
        except (TypeError, IndexError):
            Yvar = None

        subW = tk.Toplevel(M)
        subW.grab_set()
        FitPreviewWindow(subW, X, Y, reg, Yvar)
        M.wait_window(subW)

class FitPreviewWindow:
    def __init__(self, parent, X, Y, reg, Yvar=None):
        parent.title(f'Region {reg} fitted with function {reg.func}')

        popt, pcov = pk_elab._manage_fit(X, Y, reg.centroid, reg.func, Y2=Yvar)

        graph_part = tk.Frame(parent)

        if popt[0] is not None:
            self.figure = Figure(figsize=(8.5, 6))
            self.figure.patch.set_alpha(0.0)
            self.ax, self.ax_res = self.figure.subplots(nrows=2, ncols=1, sharex=True, height_ratios=[2.5,1])
            Figur = tk.Frame(graph_part)
            Figur.grid(row=0, column=0, sticky=tk.NSEW)
            self.canvas = FigureCanvasTkAgg(self.figure, master=Figur)
            self.ax_res.set_xlabel('channel / 1')
            self.figure.tight_layout()
            self.canvas.draw()
            self.canvas.get_tk_widget().configure(background=graph_part.cget('bg'))
            self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

            kw_countplot = {'marker':'', 'linestyle':'-', 'linewidth':0.5, 'color':'k'}
            kw_fit = {'marker':'', 'linestyle':'-', 'linewidth':0.75, 'color':'r'}
            kw_BKG = {'marker':'', 'linestyle':'-', 'linewidth':0.75, 'color':'#006400'}
            kw_fit_unc = {'marker':'', 'linestyle':'--', 'linewidth':0.75, 'color':'#800080'}

            self.ax.plot(X, Y, **kw_countplot)
            x_fit = np.linspace(np.min(X), np.max(X), int(len(X)*50))
            y_fit = reg.func(x_fit, *popt)
            y_BKG = reg.funcBKG(x_fit, *popt)
            self.ax.plot(x_fit, y_fit, **kw_fit)
            self.ax.plot(x_fit, y_BKG, **kw_BKG)

            #residuals
            Y_res = (reg.func(X, *popt) - Y)/np.sqrt(Y)
            fit_unc = pk_elab._fit_unc(reg.func, X, popt, pcov)
            Y_res_unc = (reg.func(X, *popt) - Y)/np.sqrt(Y + np.power(fit_unc,2))
            self.ax_res.plot(X, Y_res, **kw_fit)
            self.ax_res.plot(X, Y_res_unc, **kw_fit_unc)

            self.ax_res.axhline(y=0, **kw_countplot)#color='k', linestyle='-', linewidth=0.75)
            self.ax_res.axhline(y=-3, color='#306AC0', linestyle='-', linewidth=1.5)
            self.ax_res.axhline(y=3, color='#306AC0', linestyle='-', linewidth=1.5)

            self.ax.set_xlim(np.min(X), np.max(X))
            self.ax.set_ylim(0, np.max(Y)*1.08 + 10)

            YMAX = np.max(np.abs(Y_res))
            if YMAX < 3:
                YMAX = 3
            self.ax_res.set_ylim(-YMAX, YMAX)

            self.figure.tight_layout()
            self.canvas.draw()

        else:
            tk.Label(graph_part, text='ERROR: FIT DID NOT CONVERGE').grid(row=0, column=0, padx=10, pady=10)

        graph_part.pack(side=tk.LEFT, anchor=tk.NW, fill=tk.BOTH, expand=True)

        if popt[0] is not None:
            peakinfo_part = tk.Frame(parent)

            perr = np.sqrt(np.diag(pcov))

            #popt
            tk.Label(peakinfo_part, text='x', width=11).grid(row=0, column=1, padx=3)
            tk.Label(peakinfo_part, text='ur(x)', width=7).grid(row=0, column=2, padx=3)
            tk.Label(peakinfo_part, text='NET AREA').grid(row=1, column=0, sticky=tk.W)
            tk.Label(peakinfo_part, text='CENTROID').grid(row=2, column=0, sticky=tk.W)
            tk.Label(peakinfo_part, text='SIGMA').grid(row=3, column=0, sticky=tk.W)
            tk.Label(peakinfo_part, text='FWHM (CH)').grid(row=4, column=0, sticky=tk.W)

            tk.Label(peakinfo_part, text=f'{popt[0]:.0f}').grid(row=1, column=1)
            tk.Label(peakinfo_part, text=f'{popt[1]:.2f}').grid(row=2, column=1)
            tk.Label(peakinfo_part, text=f'{popt[2]:.2f}').grid(row=3, column=1)
            tk.Label(peakinfo_part, text=f'{popt[2]*2.355:.2f}').grid(row=4, column=1)

            tk.Label(peakinfo_part, text=f'{perr[0]/popt[0]*100:.2f} %').grid(row=1, column=2)
            tk.Label(peakinfo_part, text=f'{perr[1]/popt[1]*100:.2f} %').grid(row=2, column=2)
            tk.Label(peakinfo_part, text=f'{perr[2]/popt[2]*100:.2f} %').grid(row=3, column=2)
            tk.Label(peakinfo_part, text=f'{perr[2]/popt[2]*235.5:.2f} %').grid(row=4, column=2)

            peakinfo_part.pack(side=tk.RIGHT, anchor=tk.NW)


class OptionSelector:
    def __init__(self, M):
        M.title('Options selector')
        self.options = {}

        main_header = tk.Frame(M)

        self.funtion_CB = ttk.Combobox(main_header, state='readonly', width=50)
        self.funtion_CB['values'] = ('open_digitized_spectrumPHA', 'multi_spectrum_elab')

        tk.Label(main_header, text='function').pack()
        self.funtion_CB.pack(pady=3)
        main_header.pack(anchor=tk.N, pady=10)

        self.pages = []

        self._create_page01(M)
        self._create_page02(M)

        self.funtion_CB.bind('<<ComboboxSelected>>', lambda e='<<ComboboxSelected>>' : self._appear())
        self.funtion_CB.set(self.funtion_CB['values'][0])
        self._appear()

    def _create_page01(self, M):
        page01 = tk.LabelFrame(M, labelwidget=tk.Label(M, text='options'), relief='solid', bd=2, padx=4, pady=4)

        self.filenamePHA = None

        tk.Label(page01, text='filename', anchor=tk.W, width=self.funtion_CB.cget('width')//2).grid(row=0, column=0, sticky=tk.W)
        tk.Label(page01, text='chunksize', anchor=tk.W).grid(row=1, column=0, sticky=tk.W)
        tk.Label(page01, text='extended_deadtime', anchor=tk.W).grid(row=2, column=0, sticky=tk.W)
        tk.Label(page01, text='μs', anchor=tk.W).grid(row=2, column=2, sticky=tk.E)
        tk.Label(page01, text='pileuprejection_limit', anchor=tk.W).grid(row=3, column=0, sticky=tk.W)
        tk.Label(page01, text='μs', anchor=tk.W).grid(row=3, column=2, sticky=tk.E)
        tk.Label(page01, text='spectrum_channels', anchor=tk.W).grid(row=4, column=0, sticky=tk.W)
        tk.Label(page01, text='verbose', anchor=tk.W).grid(row=5, column=0, sticky=tk.W)
        tk.Label(page01, text='LFC', anchor=tk.W).grid(row=6, column=0, sticky=tk.W)
        tk.Label(page01, text='LFC_type', anchor=tk.W).grid(row=7, column=0, sticky=tk.W)

        fname_BT = tk.Button(page01, text='FETCH FILE', width=15)
        fname_BT.grid(row=0, column=1, sticky=tk.E)

        self.inspect_BT = tk.Label(page01, text='-', width=2)
        self.inspect_BT.grid(row=0, column=2, sticky=tk.W, padx=2)

        self.chunk_size_PHASC = ttk.Combobox(page01, state='readonly', width=12)#tk.Scale(page01, from_=100, to=5000000, resolution=100, orient=tk.HORIZONTAL, length=200, width=5)
        self.chunk_size_PHASC.grid(row=1, column=1, sticky=tk.E)
        #_values = [100 + i*100 for i in range(9)] + [1000 + i*1000 for i in range(9)] + [10000 + i*10000 for i in range(10)] + [250000 + i*250000 for i in range(20)]
        _values = [10000 + i*10000 for i in range(10)] + [250000 + i*250000 for i in range(20)]
        self.chunk_size_PHASC['values'] = _values
        self.chunk_size_PHASC.set(100000)

        self.extended_deadtime_PHAE = tk.Spinbox(page01, width=10, from_=1.0, to=250.0, increment=0.1)
        self.extended_deadtime_PHAE.grid(row=2, column=1, sticky=tk.E)
        self.extended_deadtime_PHAE.delete(0, tk.END)
        self.extended_deadtime_PHAE.insert(0, 100.0)

        self.piluuprerejectionlimit_PHAE = tk.Spinbox(page01, width=10, from_=0.0, to=100.0, increment=0.1)
        self.piluuprerejectionlimit_PHAE.grid(row=3, column=1, sticky=tk.E)
        self.piluuprerejectionlimit_PHAE.delete(0, tk.END)
        self.piluuprerejectionlimit_PHAE.insert(0, 0.5)

        self.spectrum_channels_PHASC = ttk.Combobox(page01, state='readonly', width=12)
        self.spectrum_channels_PHASC.grid(row=4, column=1, sticky=tk.E)
        _values = [2**i for i in range(11, 20, 1)]
        self.spectrum_channels_PHASC['values'] = _values
        self.spectrum_channels_PHASC.set(2**14)

        self.verbose_BVPHA = tk.BooleanVar(page01)
        CHBVERB = tk.Checkbutton(page01, text='', variable=self.verbose_BVPHA)
        CHBVERB.grid(row=5, column=1, sticky=tk.E)
        self.verbose_BVPHA.set(True)

        self.LFC_BVPHA = tk.BooleanVar(page01)
        CHBLFC = tk.Checkbutton(page01, text='', variable=self.LFC_BVPHA)
        CHBLFC.grid(row=6, column=1, sticky=tk.E)
        self.LFC_BVPHA.set(False)

        self.LFCtype_PHACB = ttk.Combobox(page01, state='readonly', width=12)
        self.LFCtype_PHACB.grid(row=7, column=1, sticky=tk.E)
        self.LFCtype_PHACB['values'] = ('ICR', 'time')
        self.LFCtype_PHACB.set('ICR')

        confirm_BT = tk.Button(page01, text='CONFIRM', width=20)
        confirm_BT.grid(row=10, column=0, columnspan=3, sticky=tk.W, pady=5)

        fname_BT.configure(command=lambda M=M : self._cmdPHA(M))
        confirm_BT.configure(command=lambda M=M : self.confirm_closePHA(M))

        self.pages.append(page01)

    def _appear(self):
        [item.pack(padx=5, pady=5, fill=tk.X, expand=True) if idx == self.funtion_CB.current() else item.pack_forget() for idx, item in enumerate(self.pages)]

    def _cmdPHA(self, M):
        self.filenamePHA = recall_filename(ftype='csv')

        if self.filenamePHA is not None and self.filenamePHA != '':
            self.inspect_BT.configure(text='$')

    def confirm_closePHA(self, M):
        if self.filenamePHA is not None and self.filenamePHA != '':
            self.options['filename'] = self.filenamePHA
            self.options['chunksize'] = int(self.chunk_size_PHASC.get())
            self.options['channel_signal'] = None
            self.options['extended_deadtime'] = float(self.extended_deadtime_PHAE.get())
            self.options['pileuprejection_limit'] = float(self.piluuprerejectionlimit_PHAE.get())
            self.options['spectrum_channels'] = int(self.spectrum_channels_PHASC.get())
            self.options['verbose'] = self.verbose_BVPHA.get()
            self.options['LFC'] = self.LFC_BVPHA.get()
            self.options['LFC_type'] = self.LFCtype_PHACB.get()

            M.destroy()
        else:
            print('select a file from the corresponding button')

    def _create_page02(self, M):
        page02 = tk.LabelFrame(M, labelwidget=tk.Label(M, text='options'), relief='solid', bd=2, padx=4, pady=4)

        self.filenamesMTE = None
        self.extended_deadtime_MTEE = [10]
        self.folderMTE = None

        tk.Label(page02, text='filenames', anchor=tk.W, width=self.funtion_CB.cget('width')//2).grid(row=0, column=0, sticky=tk.W)
        tk.Label(page02, text='chunksize', anchor=tk.W).grid(row=1, column=0, sticky=tk.W)
        tk.Label(page02, text='extended_deadtimes', anchor=tk.W).grid(row=2, column=0, sticky=tk.W)
        tk.Label(page02, text='pileuprejection_limit', anchor=tk.W).grid(row=3, column=0, sticky=tk.W)
        tk.Label(page02, text='μs', anchor=tk.W).grid(row=3, column=2, sticky=tk.E)
        tk.Label(page02, text='spectrum_channels', anchor=tk.W).grid(row=4, column=0, sticky=tk.W)
        tk.Label(page02, text='verbose', anchor=tk.W).grid(row=5, column=0, sticky=tk.W)
        tk.Label(page02, text='LFC', anchor=tk.W).grid(row=6, column=0, sticky=tk.W)
        tk.Label(page02, text='LFC_type', anchor=tk.W).grid(row=7, column=0, sticky=tk.W)
        tk.Label(page02, text='full_output', anchor=tk.W).grid(row=8, column=0, sticky=tk.W)
        tk.Label(page02, text='save_folder', anchor=tk.W).grid(row=9, column=0, sticky=tk.W)
        tk.Label(page02, text='save_format', anchor=tk.W).grid(row=10, column=0, sticky=tk.W)

        fnames_BT = tk.Button(page02, text='FETCH FILES', width=15)
        fnames_BT.grid(row=0, column=1, sticky=tk.E)

        self.inspect_BTMTE = tk.Label(page02, text='-', width=2)
        self.inspect_BTMTE.grid(row=0, column=2, sticky=tk.W, padx=2)

        self.chunk_size_MTESC = ttk.Combobox(page02, state='readonly', width=12)
        self.chunk_size_MTESC.grid(row=1, column=1, sticky=tk.E)
        _values = [10000 + i*10000 for i in range(10)] + [250000 + i*250000 for i in range(20)]
        self.chunk_size_MTESC['values'] = _values
        self.chunk_size_MTESC.set(100000)

        edts_BT = tk.Button(page02, text='DEAD_TIMES', width=15)
        edts_BT.grid(row=2, column=1, sticky=tk.E)
        self.extended_deadtime_BTMTEE = tk.Label(page02, text=f'{len(self.extended_deadtime_MTEE)}', width=2)
        self.extended_deadtime_BTMTEE.grid(row=2, column=2, sticky=tk.W)

        self.piluuprerejectionlimit_MTEE = tk.Spinbox(page02, width=10, from_=0.0, to=100.0, increment=0.1)
        self.piluuprerejectionlimit_MTEE.grid(row=3, column=1, sticky=tk.E)
        self.piluuprerejectionlimit_MTEE.delete(0, tk.END)
        self.piluuprerejectionlimit_MTEE.insert(0, 0.5)

        self.spectrum_channels_MTESC = ttk.Combobox(page02, state='readonly', width=12)
        self.spectrum_channels_MTESC.grid(row=4, column=1, sticky=tk.E)
        _values = [2**i for i in range(11, 20, 1)]
        self.spectrum_channels_MTESC['values'] = _values
        self.spectrum_channels_MTESC.set(2**14)

        self.verbose_BVMTE = tk.BooleanVar(page02)
        CHBVERBMTE = tk.Checkbutton(page02, text='', variable=self.verbose_BVMTE)
        CHBVERBMTE.grid(row=5, column=1, sticky=tk.E)
        self.verbose_BVMTE.set(True)

        self.LFC_BVMTE = tk.BooleanVar(page02)
        CHBLFCMTE = tk.Checkbutton(page02, text='', variable=self.LFC_BVMTE)
        CHBLFCMTE.grid(row=6, column=1, sticky=tk.E)
        self.LFC_BVMTE.set(False)

        self.LFCtype_MTECB = ttk.Combobox(page02, state='readonly', width=12)
        self.LFCtype_MTECB.grid(row=7, column=1, sticky=tk.E)
        self.LFCtype_MTECB['values'] = ('ICR', 'time')
        self.LFCtype_MTECB.set('ICR')

        self.full_output_BVMTE = tk.BooleanVar(page02)
        CHBLFCMTEFO = tk.Checkbutton(page02, text='', variable=self.full_output_BVMTE)
        CHBLFCMTEFO.grid(row=8, column=1, sticky=tk.E)
        self.full_output_BVMTE.set(True)

        ffolder_BT = tk.Button(page02, text='FOLDER', width=15)
        ffolder_BT.grid(row=9, column=1, sticky=tk.E)

        self.inspect_FOLD_BTMTE = tk.Label(page02, text='-', width=2)
        self.inspect_FOLD_BTMTE.grid(row=9, column=2, sticky=tk.W, padx=2)

        self.save_format_MTECB = ttk.Combobox(page02, state='readonly', width=12)
        self.save_format_MTECB.grid(row=10, column=1, sticky=tk.E)
        self.save_format_MTECB['values'] = ('n42', 'asc')
        self.save_format_MTECB.set('n42')

        confirm_BTMTE = tk.Button(page02, text='CONFIRM', width=20)
        confirm_BTMTE.grid(row=12, column=0, columnspan=3, sticky=tk.W, pady=5)

        fnames_BT.configure(command=lambda M=M : self._cmdMTE(M))
        ffolder_BT.configure(command=lambda M=M : self._cmdfldMTE(M))
        edts_BT.configure(command=lambda M=M : self._select_edts(M))
        confirm_BTMTE.configure(command=lambda M=M : self.confirm_closeMTE(M))

        self.pages.append(page02)

    def _cmdMTE(self, M):
        self.filenamesMTE = recall_filename(single=False, ftype='csv')

        if self.filenamesMTE is not None and self.filenamesMTE != ():
            self.inspect_BTMTE.configure(text=f'{len(self.filenamesMTE)}')

    def _cmdfldMTE(self, M):#folder
        self.folderMTE = select_directory()

        if self.folderMTE is not None and self.folderMTE != '':
            self.inspect_FOLD_BTMTE.configure(text='$')
        else:
            self.folderMTE = None

    def _select_edts(self, M):
        T_EDTS = tk.Toplevel(M)
        T_EDTS.grab_set()
        T_EDTS.title('EXTENDED DEAD TIMES')

        _WIDTH = 5

        edts = (1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 50, 75, 100, 125, 150, 175, 200)
        variables = [tk.BooleanVar(T_EDTS) for _ in edts]
        for var, value in zip(variables, edts):
            if value in self.extended_deadtime_MTEE:
                var.set(True)
            else:
                var.set(False)

        menuframe = tk.Frame(T_EDTS)
        row, col = 0, 0
        for idx in range(len(edts)):
            _ = tk.Checkbutton(menuframe, text=f'{edts[idx]:.1f} μs', variable=variables[idx])
            _.grid(row=idx//_WIDTH, column=idx%_WIDTH, sticky=tk.W, padx=5)

        menuframe.pack(padx=10, pady=10)

        conf_BT = tk.Button(T_EDTS, text='CONFIRM AND CLOSE', width=20)
        conf_BT.pack(pady=5)

        conf_BT.configure(command=lambda T_EDTS=T_EDTS, edts=edts, variables=variables : self._confirm_and_close_cmd(T_EDTS, edts, variables))
        M.wait_window(T_EDTS)

    def _confirm_and_close_cmd(self, T_EDTS, edts, variables):

        for EDT, VAR in zip(edts, variables):
            if VAR.get() == True:
                self.extended_deadtime_MTEE.append(EDT)

        self.extended_deadtime_MTEE = sorted(set(self.extended_deadtime_MTEE))
        self.extended_deadtime_BTMTEE.configure(text=f'{len(self.extended_deadtime_MTEE)}')

        T_EDTS.destroy()

    def confirm_closeMTE(self, M):

        if self.filenamesMTE is not None and self.filenamesMTE != () and len(self.extended_deadtime_MTEE) > 0:
            self.options['filenames'] = self.filenamesMTE
            self.options['chunksize'] = int(self.chunk_size_MTESC.get())
            self.options['extended_deadtimes'] = self.extended_deadtime_MTEE
            self.options['pileuprejection_limit'] = float(self.piluuprerejectionlimit_MTEE.get())
            self.options['n_channels'] = int(self.spectrum_channels_MTESC.get())
            self.options['verbose'] = self.verbose_BVMTE.get()
            self.options['LFC'] = self.LFC_BVMTE.get()
            self.options['LFC_type'] = self.LFCtype_MTECB.get()
            self.options['folder'] = self.folderMTE
            self.options['full_output'] = self.full_output_BVMTE.get()
            self.options['save_format'] = self.save_format_MTECB.get()

            M.destroy()
        else:
            print('select files and extended dead times from the corresponding buttons')


class SourceReservoir:
    def __init__(self):
        pass

def _open_single_fname(ftype='all'):
    """Return a filename as a string selected through a graphical interface

    Parameters
    ----------
    ftype : str
        select specific filetypes to filter

    Return
    ------
    filename : str
        filename (filepath)
    """
    M = OpenS(None, ftype=ftype)
    return str(M.filename)

def _open_multi_fname(ftype='all'):
    """Return tuple of filenames selected through a graphical interface

    Parameters
    ----------
    ftype : str
        select specific filetypes to filter

    Return
    ------
    filename : tuple
        tuple of filenames (filepaths)
    """
    M = OpenM(None, ftype=ftype)
    return tuple(M.filenames)

def select_directory():
    """Return tuple of filenames selected through a graphical interface

    Parameters
    ----------
    ftype : str
        select specific filetypes to filter

    Return
    ------
    filename : tuple
        tuple of filenames (filepaths)
    """
    M = OpenD(None)
    return M.folder

def recall_filename(single=True, ftype='all'):
    """Return filenames selected through a graphical interface

    Parameters
    ----------
    single : str (path)
        name of the gamma spectrum in list mode file
    ftypes : str
        select specific filetypes to filter

    Return
    ------
    filename : str or tuple (depending if single is True or False, respectively)
        filename or list of filenames (filepaths)
    """
    if single:
        return _open_single_fname(ftype=ftype)
    else:
        return _open_multi_fname(ftype=ftype)

def get_options():
    M = tk.Tk()
    OSW = OptionSelector(M)
    M.mainloop()

    return OSW.options

def find_elaboration_regions(spc, bkg=None):
    M = tk.Tk()
    RF = RegionFinder(M, spc, bkg)
    M.mainloop()

    return RF.spc_elab
