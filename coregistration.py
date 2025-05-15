import mne
import os
from argparse import ArgumentParser
from mne.coreg import Coregistration
from mne.io import read_info

# adapted from
# https://mne.tools/stable/auto_tutorials/forward/25_automated_coreg.html#sphx-glr-auto-tutorials-forward-25-automated-coreg-py 

def main():

    parser = ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="root data folder")
    parser.add_argument("--save_dir", type=str, required=True, help='dir to save the figures')
    parser.add_argument("--mne_dir", type=str, help="mne folder")

    args = parser.parse_args()
    root = args.root
    save_dir = args.save_dir
    mne_dir = args.mne_dir
    subjects_dir = args.mne_dir if args.mne_dir is not None else os.path.dirname(mne.datasets.fetch_fsaverage(verbose=True))

    os.environ["SUBJECTS_DIR"] = subjects_dir


    fs_average = 'fsaverage'
    fiducials = "estimated" #fsaverage fiducials
    view_kwargs = dict(azimuth=45, elevation=90, distance=0.6, focalpoint=(0.0, 0.0, 0.0))

    filecount=0
    meg_files = []
    for root, dirs, files in os.walk(root):
        for file in files:
            if file.endswith("ica_raw.fif"):
                print(f"Found FIF file: {os.path.join(root, file)}")
                meg_files.append(os.path.join(root, file))
                filecount += 1
            
        print(filecount)
        print(len(meg_files))


    for file in meg_files:
        parsed_filename = file.split("/")[8] + "_" + file.split("/")[9]

        print(f"processing subject {parsed_filename}")

        save_file_name = parsed_filename + "-trans.fif"
        wd = os.getcwd() + "/trans/"
        save_file_path = os.path.join(wd, save_file_name)
    
        # # create the info file
        info = read_info(file)
        coreg = Coregistration(
            info,
            fs_average,
            subjects_dir,
            fiducials=fiducials
        )

        # fitting
        coreg.fit_fiducials(verbose=True)
        coreg.set_scale_mode('3-axis')  # scale along all 3 axes

        #iterate
        coreg.fit_icp(n_iterations=20, nasion_weight=10.0)
        coreg.omit_head_shape_points(distance=10.0/ 1000)
        coreg.fit_icp(n_iterations=20, nasion_weight=10.0)
        coreg.omit_head_shape_points(distance=10.0/ 1000)
        coreg.fit_icp(n_iterations=20, nasion_weight=10.0)
        

        scaling_params = coreg._scale  # Access the scaling parameters
        subject_to = f'scaled_fsaverage_{parsed_filename}'  # naming for scaled MRI
        
        # get and save scaled MRI
        mne.scale_mri(
        subject_from='fsaverage',
        subject_to=subject_to,
        scale=scaling_params,
        subjects_dir=subjects_dir,
        overwrite=True
        )

        #plotting
        fig = mne.viz.plot_alignment(
            info, 
            trans=coreg.trans, 
            subject=subject_to,
            subjects_dir=subjects_dir,
            surfaces='head-dense',
            dig=True,
            eeg=[],
            meg="sensors",
            show_axes=True,
            coord_frame="meg"
        )

        mne.viz.set_3d_view(fig, **view_kwargs)

        fig_path = os.path.join(save_dir, f"{parsed_filename}_coreg.html")
        fig.plotter.export_html(fig_path)
        fig.plotter.close()


        mne.write_trans(
            save_file_path, 
            coreg.trans,
            overwrite=True
        )

        surfs = mne.read_bem_surfaces(
            os.path.join(
                f"{mne_dir}/{subject_to}/bem", f'{subject_to}-5120-5120-5120-bem.fif'
            )
        )
        bem = mne.make_bem_solution(surfs)
        mne.write_bem_solution(
            os.path.join(
                f"{mne_dir}/{subject_to}/bem", f'{subject_to}-5120-5120-5120-bem-sol.fif'
                ), 
            bem,
            overwrite=True)


if __name__ == "__main__":
    main()