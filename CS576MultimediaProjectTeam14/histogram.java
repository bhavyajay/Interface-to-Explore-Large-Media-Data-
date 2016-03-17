import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Graphics;
import java.awt.GridBagConstraints;
import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.beans.FeatureDescriptor;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.FilenameFilter;
import java.io.IOException;
import java.io.InputStream;
import java.text.DateFormat.Field;
import java.util.ArrayList;
import java.util.List;
import java.lang.*;
import java.lang.reflect.Array;

import javax.imageio.ImageIO;
import javax.swing.BorderFactory;
import javax.swing.ImageIcon;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.Timer;

import org.opencv.contrib.FaceRecognizer;
import org.opencv.core.Core;
import org.opencv.core.Core.MinMaxLocResult;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.features2d.DMatch;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.*;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.HOGDescriptor;
import org.opencv.objdetect.Objdetect;

public class histogram {


		static String foldername ="images";
		// File representing the folder that you select using a FileChooser
		static final File dir = new File(foldername);//"images");

		// array of supported extensions (use a List if you prefer)
		static final String[] EXTENSIONS = new String[] { "rgb", "png", "bmp",
				"jpg" // and other formats you need
		};

		static int width = 352;
		static int height = 288;
		// filter to identify images based on their extensions
		static final FilenameFilter IMAGE_FILTER = new FilenameFilter() {

			@Override
			public boolean accept(final File dir, final String name) {
				for (final String ext : EXTENSIONS) {
					if (name.endsWith("." + ext)) {
						return (true);
					}
				}
				return (false);
			}
		};
	static String Bhavesh[][]=null;
	static int  no_of_images= 0, maxgroup = 0;
		// read .rgb file and convert it to buffered image
		static BufferedImage readFile(String filename) {
			BufferedImage img = new BufferedImage(width, height,
					BufferedImage.TYPE_INT_RGB);

			try {
				// img = ImageIO.read(f);
				File f1 = new File(filename);
				InputStream is = new FileInputStream(f1);

				long len = f1.length();
				byte[] bytes = new byte[(int) len];

				int offset = 0;
				int numRead = 0;
				while (offset < bytes.length
						&& (numRead = is.read(bytes, offset, bytes.length - offset)) >= 0) {
					offset += numRead;
				}

				int ind = 0;
				for (int y = 0; y < height; y++) {

					for (int x = 0; x < width; x++) {

						byte a = 0;
						byte r = bytes[ind];
						byte g = bytes[ind + height * width];
						byte b = bytes[ind + height * width * 2];

						int pix = 0xff000000 | ((r & 0xff) << 16)
								| ((g & 0xff) << 8) | (b & 0xff);
						// int pix = ((a << 24) + (r << 16) + (g << 8) + b);

						img.setRGB(x, y, pix);
						ind++;
					}
				}

			} catch (final IOException e) {
			}

			return img;
		}

		// convert buffered image data to bytes
		static byte[] imageDataByte(BufferedImage img) {
			byte[] dataImg = new byte[width * height * 3];
			int[] dataBuff = img.getRGB(0, 0, width, height, null, 0, width);
			for (int i = 0; i < dataBuff.length; i++) {
				dataImg[i * 3] = (byte) ((dataBuff[i] >> 16) & 0xFF);
				dataImg[i * 3 + 1] = (byte) ((dataBuff[i] >> 8) & 0xFF);
				dataImg[i * 3 + 2] = (byte) ((dataBuff[i] >> 0) & 0xFF);
			}
			return dataImg;
		}

		static double colorComparison(Mat mat1, Mat mat2) {
			Mat mat_hsv1 = new Mat(width, height, CvType.CV_8UC3);
			Mat mat_hsv2 = new Mat(width, height, CvType.CV_8UC3);

			Imgproc.cvtColor(mat1, mat_hsv1, Imgproc.COLOR_RGB2HSV);
			Imgproc.cvtColor(mat2, mat_hsv2, Imgproc.COLOR_RGB2HSV);
			MatOfInt histSize = new MatOfInt(50, 60);
			MatOfFloat ranges = new MatOfFloat(0.0f, 180.0f, 0.0f, 256.0f);

			MatOfInt channels = new MatOfInt(0, 1);

			Mat hist1 = new Mat();
			Mat hist2 = new Mat();
			// / Calculate the histograms for the HSV images
			ArrayList<Mat> image1 = new ArrayList<Mat>();
			Core.split(mat_hsv1, image1);
			Imgproc.calcHist(image1, channels, new Mat(), hist1, histSize, ranges,
					false);
			Core.normalize(hist1, hist1, 0, hist1.rows(), Core.NORM_MINMAX, -1,
					new Mat());

			ArrayList<Mat> image2 = new ArrayList<Mat>();
			Core.split(mat_hsv2, image2);
			Imgproc.calcHist(image2, channels, new Mat(), hist2, histSize, ranges,
					false);
			Core.normalize(hist2, hist2, 0, hist2.rows(), Core.NORM_MINMAX, -1,
					new Mat());

			hist1.convertTo(hist1, CvType.CV_32F);
			hist2.convertTo(hist2, CvType.CV_32F);

			double compareHist = Imgproc.compareHist(hist1, hist2,
					Imgproc.CV_COMP_BHATTACHARYYA);

			return compareHist;
		}
		
		static double stdDev(List<DMatch> matchesList){
			double avg = 0,sum=0;
			for(int i = 0;i<matchesList.size();i++)
			{
				avg+=matchesList.get(i).distance;
			}
			avg = avg/matchesList.size();
			for(int i = 0;i<matchesList.size();i++)
			{
				sum = sum + Math.pow((matchesList.get(i).distance - avg),2);
			}
			sum = sum/matchesList.size();
			return Math.sqrt(sum);
		}

		static int featureComparison(Mat mat1, Mat mat2) {
			Mat mat_gray1 = new Mat(width, height, CvType.CV_8UC3);
			Mat mat_gray2 = new Mat(width, height, CvType.CV_8UC3);

			Imgproc.cvtColor(mat1, mat_gray1, Imgproc.COLOR_RGB2GRAY);
			Imgproc.cvtColor(mat2, mat_gray2, Imgproc.COLOR_RGB2GRAY);

			FeatureDetector fd = FeatureDetector.create(FeatureDetector.ORB);
			final MatOfKeyPoint keyPointsimg1 = new MatOfKeyPoint();
			final MatOfKeyPoint keyPointsimg2 = new MatOfKeyPoint();

			fd.detect(mat_gray1, keyPointsimg1);
			fd.detect(mat_gray2, keyPointsimg2);

			Mat descriptorsimg1 = new Mat();
			Mat descriptorsimg2 = new Mat();

			DescriptorExtractor extractor = DescriptorExtractor
					.create(DescriptorExtractor.ORB);
			extractor.compute(mat_gray1, keyPointsimg1, descriptorsimg1);
			extractor.compute(mat_gray2, keyPointsimg2, descriptorsimg2);

			MatOfDMatch matches = new MatOfDMatch();

			DescriptorMatcher matcher = DescriptorMatcher
					.create(DescriptorMatcher.BRUTEFORCE_HAMMING);
			matcher.match(descriptorsimg1, descriptorsimg2, matches);

			List<DMatch> matchesList = matches.toList();
			ArrayList<DMatch> matches_final = new ArrayList<DMatch>();
			Double max_dist = 0.0;
			Double min_dist = 70.0;
			
			double stdev = stdDev(matchesList);
//			for (int l = 0; l < matchesList.size(); l++) {
//				if (matchesList.get(l).distance <= min_dist) {
//					matches_final.add(matches.toList().get(l));
//				}
	//
//			}
//			MatOfDMatch matches_final_mat = new MatOfDMatch();
//			matches_final_mat.fromList(matches_final);
	//
//			List<DMatch> finalMatchesList = matches_final_mat.toList();
			int res = 0;
			if(stdev<0.5)
			{ res = 1;}
			return res;//finalMatchesList.size();
		}
		
		static double calculateEntropy(Mat mat1){
			
			double entropy =0.0;
			Mat m1 = new Mat();
			Imgproc.cvtColor(mat1,m1,Imgproc.COLOR_BGR2GRAY );
			
			MatOfInt histSize = new MatOfInt(256);
			MatOfFloat ranges = new MatOfFloat(0.0f, 256.0f);

			MatOfInt channels = new MatOfInt(0);

		    Mat hist1 = new Mat();
			ArrayList<Mat> image1 = new ArrayList<Mat>();
			Core.split(m1, image1);
			Imgproc.calcHist(image1, channels, new Mat(), hist1, histSize, ranges,
					false);
		   
			//Core.divide(hist1,new Scalar(m1.total()), hist1);
			double total = 1.0/m1.total();
			Core.multiply(hist1, new Scalar(total), hist1);
		    Mat logP = new Mat();
		    Core.log(hist1,logP);
		    Mat mulres = hist1.mul(logP);
		    Scalar sum = Core.sumElems(mulres);
		    Scalar e = sum.mul(new Scalar(-1));

		    entropy = e.val[0];

			return entropy;
		}
		
		static double getPSNR(Mat  I1, Mat I2)
	 {
		Mat s1 = new Mat();
		 Core.absdiff(I1, I2, s1); // |I1 - I2|
		 s1.convertTo(s1, CvType.CV_32F); // cannot make a square on 8 bits
		 s1 = s1.mul(s1); // |I1 - I2|^2
		
		Scalar s = Core.sumElems(s1); // sum elements per channel
		double sse = s.val[0] + s.val[1] + s.val[2]; // sum channels
		 if( sse <= 1e-10) // for small values return zero
		 return 0;
		 else
	 {
		 double mse = sse / (double)(I1.channels() * I1.total());
		double psnr = 10.0 * Math.log10((255 * 255) / mse);
		 return psnr;
		 }
		 }
		
		static Scalar getMSSIM(Mat i1, Mat i2)
		 {
		  double C1 = 6.5025, C2 = 58.5225;
		 /***************************** INITS **********************************/
		 int d = CvType.CV_32F;

	 Mat I1 = new Mat(), I2= new Mat();
		 i1.convertTo(I1, d); // cannot calculate on one byte large values
		 i2.convertTo(I2, d);
		
		 Mat I2_2 = I2.mul(I2); // I2^2
		 Mat I1_2 = I1.mul(I1); // I1^2
		 Mat I1_I2 = I1.mul(I2); // I1 * I2

		 Size s = new Size(11,11);
		 /*************************** END INITS **********************************/
		
		 Mat mu1= new Mat(), mu2= new Mat(); // PRELIMINARY COMPUTING
		 Imgproc.GaussianBlur(I1, mu1, s, 1.5);
		 Imgproc.GaussianBlur(I2, mu2, s, 1.5);
		
		 Mat mu1_2 = mu1.mul(mu1);
		 Mat mu2_2 = mu2.mul(mu2);
		Mat mu1_mu2 = mu1.mul(mu2);
		
		 Mat sigma1_2= new Mat(), sigma2_2= new Mat(), sigma12= new Mat();
		
		 Imgproc.GaussianBlur(I1_2, sigma1_2, s, 1.5);
		 
		 Core.subtract(sigma1_2, mu1_2, sigma1_2); //sigma1_2 -= mu1_2;
		
		 Imgproc.GaussianBlur(I2_2, sigma2_2, s, 1.5);
		 Core.subtract(sigma2_2, mu2_2, sigma2_2);//sigma2_2 -= mu2_2;
		
		 Imgproc.GaussianBlur(I1_I2, sigma12, s, 1.5);
		 Core.subtract(sigma12, mu1_mu2, sigma12);//sigma12 -= mu1_mu2;
		 ///////////////////////////////// FORMULA ////////////////////////////////
		 Mat t1= new Mat(), t2= new Mat(), t3= new Mat();
		
		 Core.multiply(mu1_mu2,new Scalar(2), t1);//t1 = 2 * mu1_mu2 + C1;
		 Core.add(t1, new Scalar(C1), t1);

		
		 Core.multiply(sigma12,new Scalar(2), t2);//	t2 = 2 * sigma12 + C2;
		 Core.add(t2, new Scalar(C2), t2);
		 t3 = t1.mul(t2); // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))
		
		 Core.add(mu1_2, mu2_2, t1); //t1 = mu1_2 + mu2_2 + C1;
		 Core.add(t1, new Scalar(C1), t1);
		 Core.add(sigma1_2, sigma2_2, t2); 
		 Core.add(t2, new Scalar(C2), t2);// t2 = sigma1_2 + sigma2_2 + C2;
		 t1 = t1.mul(t2); // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))
		
		 Mat ssim_map= new Mat();
		 Core.divide(t3, t1, ssim_map); // ssim_map = t3./t1;
		
		Scalar mssim = Core.mean(ssim_map); // mssim = average of ssim map
		return mssim;
		}
		
		static BufferedImage[] readVideo(String filename) {
			BufferedImage img[] = null;
			try {
				int index = 0;
				File file = new File(filename);
				InputStream is = new FileInputStream(file);

				double len = file.length();
				byte[] bytes = new byte[(int) len];

				int offset = 0;
				int numRead = 0;

				 int startFrame =6;
				int frames =  (int) ((len / (width * height * 3))/startFrame);
				//System.out.println(frames);
				img = new BufferedImage[frames];
				for (int i = 0; i < img.length; i++) {
					//System.out.println(i);
					img[i] = new BufferedImage(width, height,
							BufferedImage.TYPE_INT_RGB);
				}

				while (offset < bytes.length
						&& (numRead = is.read(bytes, offset, bytes.length - offset)) >= 0) {
					offset += numRead;
				}
				int ind = width * height * 3 *(startFrame);
				// for (int f = 0; f < bytes.length; f+=height*width*3) {
				for (int iter = 0; iter < frames; iter++) {
					
					for (int y = 0; y < height; y++) {

						for (int x = 0; x < width; x++) {

							byte a = 0;
							byte r = bytes[ind];
							byte g = bytes[ind + height * width];
							byte b = bytes[ind + height * width * 2];

							int pix = 0xff000000 | ((r & 0xff) << 16)
									| ((g & 0xff) << 8) | (b & 0xff);
							// int pix = ((a << 24) + (r << 16) + (g << 8) + b);
							img[iter].setRGB(x, y, pix);
							ind++;
						}
					}
					startFrame = startFrame + 5;
					ind = width * height * 3 * (startFrame);
				}
			} catch (Exception e) {

			}
			return img;
		}
		
		static BufferedImage[] readVideoComplete(String filename) {
			BufferedImage img[] = null;
			try {
				int index = 0;
				File file = new File(filename);
				InputStream is = new FileInputStream(file);

				double len = file.length();
				byte[] bytes = new byte[(int) len];

				int offset = 0;
				int numRead = 0;

				int frames =  (int) ((len / (width * height * 3)));
				//System.out.println(frames);
				img = new BufferedImage[frames];
				for (int i = 0; i < img.length; i++) {
					//System.out.println(i);
					img[i] = new BufferedImage(width, height,
							BufferedImage.TYPE_INT_RGB);
				}

				while (offset < bytes.length
						&& (numRead = is.read(bytes, offset, bytes.length - offset)) >= 0) {
					offset += numRead;
				}
				int ind = 0;// width * height * 3 *(startFrame);
				// for (int f = 0; f < bytes.length; f+=height*width*3) {
				for (int iter = 0; iter < frames; iter++) {
					
					for (int y = 0; y < height; y++) {

						for (int x = 0; x < width; x++) {

							byte a = 0;
							byte r = bytes[ind];
							byte g = bytes[ind + height * width];
							byte b = bytes[ind + height * width * 2];

							int pix = 0xff000000 | ((r & 0xff) << 16)
									| ((g & 0xff) << 8) | (b & 0xff);
							// int pix = ((a << 24) + (r << 16) + (g << 8) + b);
							img[iter].setRGB(x, y, pix);
							ind++;
						}
					}
					ind = width * height * 3 * (iter);
				}
			} catch (Exception e) {

			}
			return img;
		}

		
		static BufferedImage readVideoFrame(String filename,int frame) {
			BufferedImage imgFrame = new BufferedImage(width, height,
					BufferedImage.TYPE_INT_RGB);
			try {
				int index = 0;
				File file = new File(filename);
				System.out.println(file.exists());
				InputStream is = new FileInputStream(file);

				double len = file.length();
				byte[] bytes = new byte[(int) len];

				int offset = 0;
				int numRead = 0;
				int startFrame = 6;
				while (offset < bytes.length
						&& (numRead = is.read(bytes, offset, bytes.length - offset)) >= 0) {
					offset += numRead;
				}
				int ind = width * height * 3 *(startFrame + frame);
				// for (int f = 0; f < bytes.length; f+=height*width*3) {
					
					for (int y = 0; y < height; y++) {

						for (int x = 0; x < width; x++) {

							byte a = 0;
							byte r = bytes[ind];
							byte g = bytes[ind + height * width];
							byte b = bytes[ind + height * width * 2];

							int pix = 0xff000000 | ((r & 0xff) << 16)
									| ((g & 0xff) << 8) | (b & 0xff);
							// int pix = ((a << 24) + (r << 16) + (g << 8) + b);
							imgFrame.setRGB(x, y, pix);
							ind++;
						}
					}
				
			} catch (Exception e) {

			}
			return imgFrame;
		}

		
		static ArrayList<String[]> videoComparison(BufferedImage img[],String filename)
		{
			int groupLen = 0, eleLen = 0;
			//int no_of_images = img.length;
			byte[] dataImg1 = new byte[width * height * 3];
			byte[] dataImg2 = new byte[width * height * 3];
			String[][] imagesLookUpArray = new String[img.length][img.length];
			Mat mat1 = new Mat(width, height, CvType.CV_8UC3);
			Mat mat2 = new Mat(width, height, CvType.CV_8UC3);

			
			for (int f = 0; f < img.length; f++) {
				// adding query image to group
				dataImg1 = imageDataByte(img[f]);
				//System.out.println(files.get(f));
				imagesLookUpArray[f][0] = String.valueOf(f);//files.get(f);

				eleLen = 1;
				for (int w = 0; w < img.length; w++) {
					if (w != f) {
						dataImg2 = imageDataByte(img[w]);

						mat1.put(0, 0, dataImg1);
						mat2.put(0, 0, dataImg2);

						double colorHistThreshold = colorComparison(mat1, mat2);
				        //System.out.println("frame:"+w);
						//System.out.println(colorHistThreshold);
						
						if (colorHistThreshold < 0.2) {
							//adding different keyframe
								imagesLookUpArray[f][eleLen] = String.valueOf(w);
								//System.out.println("group:"+f+" element:"+eleLen+" " +imagesLookUpArray[f][eleLen]);
								eleLen++;
						}
					}
				}
			}
			
			//grouping key frames of video together
			int flag = 0;
		 ArrayList<String[]> discoverd = new ArrayList<String[]>();
			groupLen =0;
			for(int f = 0;f<img.length; f++){
				eleLen=0;
				for (int w = 0; w < img.length; w++) {
					if(imagesLookUpArray[f][w] != null){
				for(int i =0; i<discoverd.size();i++){
					if(discoverd.get(i) != null){
					String[] value = discoverd.get(i);
					if( value[2].equals(imagesLookUpArray[f][w])){
					groupLen = Integer.parseInt(value[1]);
					flag=1;
					}
					}
				}
				if(flag==0){
				 String[] value  = new String[3];
			      value[0] = filename;
			      value[1] = String.valueOf(groupLen);
			      value[2] = imagesLookUpArray[f][w];
				 discoverd.add(value);
				}
				flag=0;
					}
				}
				groupLen++;
			}
			ArrayList<String[]> keyFrame = new ArrayList<String[]>();
			//get only max of 5 key frames 
			for(int j = 0;j<5;j++){
				for(int i=0;i<discoverd.size();i++){
					String[] value = discoverd.get(i);
					if(j == Integer.valueOf(value[1]))
					{
						String[] value1  = new String[3];
					      value1[0] = value[0];
					      value1[1] = value[1];
					      value1[2] = value[2];
						 keyFrame.add(value);
					break;
					}
				} 
				}
			
			return keyFrame;
		}
		 static int loopF = 0;
		 static Timer timer;
		static void displayVideo(BufferedImage vid[]) throws InterruptedException
		{
			BufferedImage img = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
			JFrame frame = new JFrame();
			int frames = vid.length;
			System.out.println(frames);
			loopF = 0;	 
			boolean b;   // for starting and stoping animation
		
			    	ActionListener action = new ActionListener() {
					    @Override
					    public void actionPerformed(ActionEvent ae) {
					    	for (int i = 0; i < height; i++) {
								for (int j = 0; j < width; j++) {

									int pix1 = vid[loopF].getRGB(j, i);
									img.setRGB(j, i, pix1);
								}
							}
				JLabel label3 = new JLabel(new ImageIcon(img));
				frame.getContentPane().add(label3, BorderLayout.CENTER);
				frame.pack();
				frame.setVisible(true);
				loopF++;
				if(loopF == frames)
				{timer.stop();}
			    }
			    };
			   //	timer = new Timer(100,action );
				    
//				if ((end - start) < 50) {
//					
//					Thread.sleep(50 - (end - start));
//				}
//				        moveNext(buffer.getGraphics());
	
		if(loopF!=frames)
		{
			 timer = new Timer(50, action);
		        timer.setInitialDelay(0);
		        timer.start();
		}
		}
		
	 static BufferedImage mat2Img(Mat in)
	    {
	        BufferedImage out;
	        byte[] data = new byte[width * height * (int)in.elemSize()];
	        int type;
	        in.get(0, 0, data);

	        if(in.channels() == 1)
	            type = BufferedImage.TYPE_BYTE_GRAY;
	        else
	            type = BufferedImage.TYPE_3BYTE_BGR;

	        out = new BufferedImage(width, height, type);

	        out.getRaster().setDataElements(0, 0, width, height, data);
	        return out;
	    } 
	 
	 static void displayImage(BufferedImage img)
	 {
		 JFrame  frame1 = new JFrame();

		 frame1.setLocation(600, 0);
		 JLabel label1 = new JLabel(new ImageIcon(img));
		 frame1.getContentPane().add(label1, BorderLayout.CENTER);
		 frame1.pack();
		 frame1.setVisible(true);
	 }
	 
	 static ArrayList<String[]> imageEntropyCompare(int groupLen){

		 //adding low entropy images to group 0;
		 ArrayList<String[]>entropyGroup = new ArrayList<String[]>();
		 BufferedImage img1 = new BufferedImage(width, height,
					BufferedImage.TYPE_INT_RGB);
			BufferedImage img2 = new BufferedImage(width, height,
					BufferedImage.TYPE_INT_RGB);
			byte[] dataImg1 = new byte[width * height * 3];
			byte[] dataImg2 = new byte[width * height * 3];
			System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

			Mat mat1 = new Mat(width, height, CvType.CV_8UC3);			
			for (int f = 0; f < files.size(); f++) {
				if(files.get(f).toLowerCase().contains(String.valueOf("Image").toLowerCase())){
				// adding query image to group
				img1 = readFile(dir.getPath() + "/" + files.get(f));
				dataImg1 = imageDataByte(img1);
				//System.out.println(files.get(f));
				mat1.put(0, 0, dataImg1);
				double entropyMat = calculateEntropy(mat1);
			//	System.out.println("entropy of "+files.get(f)+"="+entropyMat);
				if(entropyMat<5)
				{
					String[] value  = new String[3];
					value[0] = files.get(f);
					value[1] = String.valueOf(groupLen);
					value[2] = "-100";
					entropyGroup.add(value);
					files.remove(f);
					f--;
				}
				
				}
			}

			for (int l = 0; l < videoKeyFrames.size(); l++) {
				String[] value =new String[3];
				value = videoKeyFrames.get(l);
				img1 = readVideoFrame(dir.getPath()+"/"+value[0],Integer.valueOf(value[2]));
				dataImg1 = imageDataByte(img1);
				mat1.put(0, 0, dataImg1);
				double entropyMat = calculateEntropy(mat1);

			//	System.out.println("entropy of "+value[0]+"="+entropyMat);
				if(entropyMat<5)
				{
					value[0] = value[0];
					value[1] = String.valueOf(groupLen);;
					value[2] = value[2];
					entropyGroup.add(value);
					videoKeyFrames.remove(l);
					l--;
				}
			}
return entropyGroup;
	 }
	 
	 static boolean calculateIsFaceDetect(Mat mat1)
	 {
			Mat m1 = new Mat();
			Imgproc.cvtColor(mat1,m1,Imgproc.COLOR_BGR2GRAY);
			
			  //BufferedImage face1 = mat2Img(m1);
        	  //displayImage(face1);
		    //Mat hist1 = new Mat();

		    //Imgproc.equalizeHist(m1, hist1);
		   // CascadeClassifier cascade = new CascadeClassifier("C:/Users/Sanskriti/Downloads/opencv/sources/data/lbpcascades/lbpcascade_frontalface.xml");
		    CascadeClassifier cascade = new CascadeClassifier("C:/Users/Sanskriti/Downloads/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml");
		    //CascadeClassifier cascadeEye = new CascadeClassifier("C:/Users/Sanskriti/Downloads/opencv/sources/data/haarcascades/haarcascade_eye.xml");
		    MatOfRect faces = new MatOfRect();
	          //if (cascade != null)
	        	cascade.detectMultiScale(m1, faces);//, 3, 1,1,new Size(30,30), new Size(30,30));
	               //cascade.detectMultiScale(hist1, faces, 1.1, 2, 0|Objdetect.CASCADE_SCALE_IMAGE, new Size(30, 30), new Size());
	          Rect[] facesArray = faces.toArray();
	          for (int i = 0; i < facesArray.length; i++)
	                  {Core.rectangle(m1, facesArray[i].tl(), facesArray[i].br(),new Scalar(255, 0, 0), 3);
	                  }
	          
	          if(facesArray.length > 0)
	          {
	        	  BufferedImage face = mat2Img(m1);
	        	  displayImage(face);
	        	  return true;
	          }
	          else
	        	  return false;
	        			 
	 }
	 
	 static ArrayList<String[]> imageFaceCompare(int groupLen){

		 //adding low entropy images to group 0;
		 ArrayList<String[]>faceGroup = new ArrayList<String[]>();
		 BufferedImage img1 = new BufferedImage(width, height,
					BufferedImage.TYPE_INT_RGB);
			BufferedImage img2 = new BufferedImage(width, height,
					BufferedImage.TYPE_INT_RGB);
			byte[] dataImg1 = new byte[width * height * 3];
			byte[] dataImg2 = new byte[width * height * 3];
			System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

			Mat mat1 = new Mat(width, height, CvType.CV_8UC3);			
			int eleLen =0;
			for (int f = 0; f < files.size(); f++) {
				if(files.get(f).toLowerCase().contains(String.valueOf("Image").toLowerCase())){
				// adding query image to group
				img1 = readFile(dir.getPath() + "/" + files.get(f));
				dataImg1 = imageDataByte(img1);
				//System.out.println(files.get(f));
				mat1.put(0, 0, dataImg1);
				boolean isFaceDetect = calculateIsFaceDetect(mat1);
				if(isFaceDetect)
				{
					String[] value  = new String[3];
					value[0] = files.get(f);
					value[1] = String.valueOf(groupLen);
					value[2] = "-100";
					faceGroup.add(value);
					files.remove(f);
				}
				
				}
			}

			for (int l = 0; l < videoKeyFrames.size(); l++) {
				String[] value =new String[3];
				value = videoKeyFrames.get(l);
				img1 = readVideoFrame(dir.getPath()+"/"+value[0],Integer.valueOf(value[2]));
				dataImg1 = imageDataByte(img1);
				mat1.put(0, 0, dataImg1);
				boolean isFaceDetect = calculateIsFaceDetect(mat1);

				if(isFaceDetect)
				{
					value[0] = value[0];
					value[1] = String.valueOf(groupLen);
					value[2] = value[2];
					faceGroup.add(value);
					videoKeyFrames.remove(l);
				}
			}
return faceGroup;
	 }
	static ArrayList<String[]> videoKeyFrames= new ArrayList<String[]>();
	static 	ArrayList<String> files = new ArrayList<String>();
		public static void main(String[] args) throws InterruptedException {
			// TODO Auto-generated method stub
			// declaration
			foldername = args[0];
			JFrame frame1 = new JFrame();
			JFrame frame2 = new JFrame();
			int groupLen = 0, eleLen = 0;
			ArrayList<String> filesCopy = new ArrayList<String>();
		
			String[][] imagesLookUpArray = null;//new String[files.size()][files.size()];
			BufferedImage img1 = new BufferedImage(width, height,
					BufferedImage.TYPE_INT_RGB);
			BufferedImage img2 = new BufferedImage(width, height,
					BufferedImage.TYPE_INT_RGB);
			byte[] dataImg1 = new byte[width * height * 3];
			byte[] dataImg2 = new byte[width * height * 3];

			System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

			Mat mat1 = new Mat(width, height, CvType.CV_8UC3);
			Mat mat2 = new Mat(width, height, CvType.CV_8UC3);

			// reading all the files from the directory
			if (dir.isDirectory()) {

				for (final File f1 : dir.listFiles(IMAGE_FILTER)) {
					files.add(f1.getName());
					filesCopy.add(f1.getName());
				}
				
			}
			
			//checking for videos
					
					for (int f = 0; f < files.size(); f++) {
						if(files.get(f).toLowerCase().contains(String.valueOf("video").toLowerCase())){
					
					BufferedImage vid[] = readVideo(dir.getPath()+"/"+files.get(f));
				//	displayVideo(vid);
					videoKeyFrames.addAll(videoComparison(vid, files.get(f)));
						}
						}
					for(int i=0;i<videoKeyFrames.size();i++){
						String[] value = videoKeyFrames.get(i);
						
					/*    System.out.println(value[0]);
					    System.out.println(value[1]);
					    System.out.println(value[2]);*/
					}
					imagesLookUpArray = new String[files.size()+videoKeyFrames.size()][files.size()+videoKeyFrames.size()];
					
				
			//first separating cartoon from real on the basis of entropy in group zero
					ArrayList<String[]> entropyGroup =  imageEntropyCompare(groupLen);
					
					if(entropyGroup.size() != 0)
					{groupLen = 1;}
					//first separating faces from images on the basis of entropy in group zero
					ArrayList<String[]> faceGroup =  imageFaceCompare(groupLen);
					
			//images
			// looping over files
					int f = 0;
			for (f = 0; f < files.size(); f++) {
				if(files.get(f).toLowerCase().contains(String.valueOf("Image").toLowerCase())){
				// adding query image to group
				img1 = readFile(dir.getPath() + "/" + files.get(f));
				dataImg1 = imageDataByte(img1);
				//System.out.println(files.get(f));
				imagesLookUpArray[f][0] = files.get(f);

				eleLen = 1;
				for (int w = f; w < files.size(); w++) {
					if (w != f && files.get(w).toLowerCase().contains(String.valueOf("Image").toLowerCase())) {
						img2 = readFile(dir.getPath() + "/" + files.get(w));
						dataImg2 = imageDataByte(img2);

						mat1.put(0, 0, dataImg1);
						mat2.put(0, 0, dataImg2);
//						Mat result = new Mat();
//						Mat mgray1 = new Mat();
//						Mat mgray2 = new Mat();
//						Imgproc.cvtColor(mat1, mgray1, Imgproc.COLOR_RGB2GRAY);
//						Imgproc.cvtColor(mat2, mgray2, Imgproc.COLOR_RGB2GRAY);
//						
//						double psnr = getPSNR(mat1, mat2);
//						System.out.println(psnr);
//						
//						Scalar ssim = getMSSIM(mat1, mat2);
//						System.out.println(" R " + ssim.val[2] * 100 +"%");
//						System.out.println(" G " + ssim.val[1] * 100 +"%");
//						System.out.println(" B " + ssim.val[0] * 100 +"%");
						//Core.compare(mgray1, mgray2, result, Imgproc.CV_COMP_BHATTACHARYYA);
				 //BufferedImage resultImage =  mat2Img(result);
				 //displayImage(resultImage);
						// compute color histograms
						double colorHistThreshold = colorComparison(mat1, mat2);
				        //System.out.println(files.get(w));
						//System.out.println(colorHistThreshold);
						
						if (colorHistThreshold >= 0 && colorHistThreshold < 0.55) {
							// do feature matching
//							int matchesFound = featureComparison(mat1, mat2);
//							System.out.println("matches found:" + matchesFound);	
//							if (matchesFound ==1) {
								imagesLookUpArray[f][eleLen] = files.get(w);
								eleLen++;
							//}
						}
					}
				}
				for(int w = 0;w<videoKeyFrames.size();w++)
				{
					String[] value =new String[3];
					value = videoKeyFrames.get(w);
					img2 = readVideoFrame(dir.getPath()+"/"+value[0],Integer.valueOf(value[2]));
					dataImg2 = imageDataByte(img2);

					mat1.put(0, 0, dataImg1);
					mat2.put(0, 0, dataImg2);
					
					// compute color histograms
					double colorHistThreshold = colorComparison(mat1, mat2);
			        //System.out.println(files.get(w));
					//System.out.println(colorHistThreshold);
					
					if (colorHistThreshold >= 0 && colorHistThreshold < 0.55) {
						imagesLookUpArray[f][eleLen] = value[0];
						eleLen++;
					}
				}
				}
			}
	//System.out.println(f);
			f = f - 3;

	//System.out.println(f);
			for (int l = 0; l < videoKeyFrames.size(); l++) {
				String[] value =new String[3];
				value = videoKeyFrames.get(l);
					
				img1 = readVideoFrame(dir.getPath()+"/"+value[0],Integer.valueOf(value[2]));
				dataImg1 = imageDataByte(img1);
				
				imagesLookUpArray[f][0] = value[0];

	eleLen =1;
			for(int w = l+1;w<videoKeyFrames.size();w++)
			{
				value = videoKeyFrames.get(w);
				img2 = readVideoFrame(dir.getPath()+"/"+value[0],Integer.valueOf(value[2]));
					
				dataImg2 = imageDataByte(img2);

				mat1.put(0, 0, dataImg1);
				mat2.put(0, 0, dataImg2);
				
				// compute color histograms
				double colorHistThreshold = colorComparison(mat1, mat2);
		        //System.out.println(files.get(w));
				//System.out.println(colorHistThreshold);
				
				if (colorHistThreshold >= 0 && colorHistThreshold < 0.55) {
					imagesLookUpArray[f][eleLen] = value[0];
					eleLen++;
				}
			}
			f++;
				}
			
			
			if(entropyGroup.size() != 0 && faceGroup.size() != 0)
			{groupLen = 2;}
			else if(entropyGroup.size() != 0 && faceGroup.size() == 0){
				groupLen = 1;	
			}
			else 
			{groupLen = 0;}

			//grouping images and video frames together
					int flag = 0;
				 ArrayList<String[]> discoverd = new ArrayList<String[]>();
					for(f = 0;f<files.size()+videoKeyFrames.size(); f++){
						eleLen=0;
						for (int w = 0; w < files.size()+videoKeyFrames.size(); w++) {
							if(imagesLookUpArray[f][w] != null){
						for(int i = 0; i<discoverd.size();i++){
							if(discoverd.get(i) != null){
							String[] value = discoverd.get(i);
							if(value[0] == imagesLookUpArray[f][w]){
							groupLen = Integer.parseInt(value[1]);
							flag=1;
							}
							}
						}
						if(flag==0){
						 String[] value  = new String[3];
						 if(imagesLookUpArray[f][w].toLowerCase().contains(String.valueOf("Image").toLowerCase()) ){
					      value[0] = imagesLookUpArray[f][w];
					      value[1] = String.valueOf(groupLen);
					      value[2] = "-100";
						 }
						 else
						 {
							 for(int k = 0;k<videoKeyFrames.size();k++)
								{
									String[] valueVid =new String[3];
									valueVid = videoKeyFrames.get(k);
								if(imagesLookUpArray[f][w].toLowerCase().equals(valueVid[0].toLowerCase())){
									value[0] = imagesLookUpArray[f][w];
									value[1] = String.valueOf(groupLen);
									value[2] = valueVid[2];
								}
								}
							 
						 }
						 discoverd.add(value);
						}
						flag=0;
							}
						}
						groupLen++;
					}
					
ArrayList<String[]> finalGroups = new ArrayList<String[]>();

					 finalGroups.addAll(entropyGroup);
					 finalGroups.addAll(faceGroup);
					 finalGroups.addAll(discoverd);
					 
					 int finalNoOfImg = finalGroups.size();
						no_of_images= finalNoOfImg;//files.size()+videoKeyFrames.size();  

						Bhavesh=new String[no_of_images][3];

						for(int i=0;i<finalNoOfImg;i++){
							String[] value = finalGroups.get(i);
							
						    System.out.println(value[0]);
						    System.out.println(value[1]);
						    System.out.println(value[2]);
						} 
					
//					discoverd.addAll(videoKeyFrames);
					System.out.println(groupLen);
					for(int i=0;i<finalNoOfImg;i++){
						String[] value = finalGroups.get(i);
						
					    System.out.println(value[0]);
					    System.out.println(value[1]);
					    System.out.println(value[2]);
					}
			
			
			
			for(int i=0;i<finalNoOfImg;i++){
				String[] value = finalGroups.get(i);
				Bhavesh[i][0]=value[1];
				Bhavesh[i][1]=value[0];
				Bhavesh[i][2] = value[2];
					if(maxgroup < Integer.valueOf(value[1]))
							{
						maxgroup = Integer.valueOf(value[1]);
							}
			   // System.out.println(i+" " +Bhavesh[i][1]);
			    //System.out.println(Bhavesh[i][0]+" ");
			} 
			
			String temp[][]=new String[finalNoOfImg][3];
			int count_er=0;
			System.out.println((maxgroup) +" "+ finalNoOfImg);
			for(int i=0;i<maxgroup+1;i++)
			{
				for(int j=0;j<finalNoOfImg;j++)
				{
					if(Bhavesh[j][0].equals(Integer.toString(i)))
					{
						temp[count_er][0]=Bhavesh[j][0];
						temp[count_er][1]=Bhavesh[j][1];
						temp[count_er][2]=Bhavesh[j][2];
						System.out.println(temp[count_er][0]+" "+temp[count_er][1]+" "+count_er);
						count_er++;
						
					}
					
				}
			}
			
			form_collage( temp, 1, finalNoOfImg, maxgroup+1);
			
//	     
			
		}
		
		public static void form_collage( String[][] Bhavesh, int level, int no_of_images, int no_of_groups ) throws InterruptedException
	    {
	    	if(level==1)
	    	{
	    	int width = 352;
	    	int height = 288;
	    	int counter=1,j=1,ptr=1;
	    	
	    	 JFrame frame = new JFrame();
	    	 frame.setSize(500, 500);
	    	 
	        while(counter<=no_of_groups)
	    	{
	                	j=ptr;
	               
	        	 BufferedImage img = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
	        	 String fileName= Bhavesh[j-1][1];
	        	 int frame_no=Integer.parseInt(Bhavesh[j-1][2]);
	            try {
	            	
	            	if(fileName.contains("image"))
	            	{		
	            	File ff= new File(dir.getPath()+"/"+fileName);
	            	InputStream is = new FileInputStream(ff);
	            	
	         	    long len = ff.length();
	         	    byte[] bytes = new byte[(int)len];
	         	    
	         	    int offset = 0;
	                int numRead = 0;
	                while (offset < bytes.length && (numRead=is.read(bytes, offset, bytes.length-offset)) >= 0) {
	                     offset += numRead;
	                 }
	                 
	                 
	             	
	             	int ind = 0;
	         		for(int y = 0; y < height; y++){
	         	
	         			for(int x = 0; x < width; x++){
	         		 
	         				byte a = 0;
	         				byte r = bytes[ind];
	         				byte g = bytes[ind+height*width];
	         				byte b = bytes[ind+height*width*2]; 
	         				
	         				int pix = 0xff000000 | ((r & 0xff) << 16) | ((g & 0xff) << 8) | (b & 0xff);
	         	
	         				img.setRGB(x,y,pix);
	         				ind++;
	         			}
	         		}
	         		
	         		
	            }
	            	else
	            	{
	            		
	            		
	            		 File file = new File(dir.getPath()+"/"+fileName);
	         		    InputStream is = new FileInputStream(file);

	         		    long len = file.length();
	         		//    System.out.println(len);
	         		    byte[] bytes = new byte[(int)len];
	         		  
	         		    
	         		   
	         		    
	         		    int offset = 0;
	         	        int numRead = 0;
	         	        
	         	        while (offset < bytes.length && (numRead=is.read(bytes, offset, bytes.length-offset)) >= 0) 
	         	        {
	         	            offset += numRead;
	         	        }
	         	    
	         	        
	         	        for(int k=frame_no; k<(frame_no+1);k++)
	         	        { 
	         	        	long startTime = System.currentTimeMillis();
	         	        int ind=352*288*3*k;
	         	        
	         	       
	         	       for(int y = 0; y < height; y++){
	         	         	
	            			for(int x = 0; x < width; x++){
	         	        			byte a = 0;
	         	    				byte r = bytes[ind];
	         	    				byte g = bytes[ind+height*width];
	         	    				byte b = bytes[ind+height*width*2]; 
	         	    				
	         	    				int pix = 0xff000000 | ((r & 0xff) << 16) | ((g & 0xff) << 8) | (b & 0xff);
	         	    				//int pix = ((a << 24) + (r << 16) + (g << 8) + b);
	         	                	img.setRGB(x, y, pix);
	         	                	ind++;
	         	                }
	         	            }
	         	        	
	                     
	         	        
	         	   }
	         			
	            		
	            		
	            		
	            	}
	          
	                System.out.println(Bhavesh[j-1][0]+" "+Bhavesh[j-1][1]);
	            
	            } catch (final IOException e) {
	               
	            }
	            
	            int dim = (int) Math.sqrt(no_of_groups);
	            GridLayout layout = new GridLayout(dim,dim);
	            GridBagConstraints c = new GridBagConstraints();
	       
	            JButton button = new JButton(new ImageIcon(img));
	           
	            button.setBorder(BorderFactory.createBevelBorder(12));
	            frame.setLayout(layout);
	            frame.getContentPane().add(button);
	            frame.pack();
	            frame.setVisible(true);
	         
	            button.addMouseListener(new java.awt.event.MouseAdapter() {
	                public void mouseEntered(java.awt.event.MouseEvent evt) {
	                
	                	button.setBorder(BorderFactory.createLineBorder(Color.black));
	                	button.setBorderPainted(true);
	                	button.setContentAreaFilled(false);
	                }
	                public void mouseExited(java.awt.event.MouseEvent evt) {
	                	button.setRolloverEnabled(true);
	                	button.setBorderPainted(false);
	                }
	            });
	            button.addActionListener(new ActionListener() 
	            {
	                public void actionPerformed(ActionEvent e) {
	                   
	                	JFrame frame2 = new JFrame();
	                	frame2.setTitle("Collage 2");
	                	
	                	//System.out.println(Bhavesh[0][1]+" "+fileName);
	                	int i=0;
	                	while(!Bhavesh[i][1].equals(fileName) & i<no_of_images)
	                	{
	                		i++;
	                	} //System.out.println("i"+i);
	                	int j=i;
	                	while(Bhavesh[j][0].equals(Bhavesh[i][0]) & j<no_of_images)
	                	{
	                		String fileName2=Bhavesh[j][1];
	                		int frameno=Integer.parseInt(Bhavesh[j][2]);
	                		BufferedImage img2 = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
	                		try {
	                        if(fileName2.contains("image"))
	                        {
	                        	
	                        	File ff2= new File(dir.getPath()+"/"+fileName2);
	                        	InputStream is = new FileInputStream(ff2);
	                        	
	                     	    long len = ff2.length();
	                     	    byte[] bytes = new byte[(int)len];
	                     	    
	                     	    int offset = 0;
	                            int numRead = 0;
	                            while (offset < bytes.length && (numRead=is.read(bytes, offset, bytes.length-offset)) >= 0) {
	                                 offset += numRead;
	                             }
	                             
	                             
	                         	
	                         	int ind = 0;
	                     		for(int y = 0; y < height; y++){
	                     	
	                     			for(int x = 0; x < width; x++){
	                     		 
	                     				byte a = 0;
	                     				byte r = bytes[ind];
	                     				byte g = bytes[ind+height*width];
	                     				byte b = bytes[ind+height*width*2]; 
	                     				
	                     				int pix = 0xff000000 | ((r & 0xff) << 16) | ((g & 0xff) << 8) | (b & 0xff);
	                     	
	                     				img2.setRGB(x,y,pix);
	                     				ind++;
	                     			}
	                     		}
	                     		
	                		}
	                		else
	                		{
	                			//System.out.println("whyyyyyyyyyy");
	                			System.out.println(frameno);

	                   		 File file = new File(dir.getPath()+"/"+fileName2);
	                   		 System.out.println(fileName2);
	                		    InputStream is = new FileInputStream(file);

	                		    long len = file.length();
	                		//    System.out.println(len);
	                		    byte[] bytes = new byte[(int)len];
	                		  
	                		    
	                		   
	                		    
	                		    int offset = 0;
	                	        int numRead = 0;
	                	        
	                	        while (offset < bytes.length && (numRead=is.read(bytes, offset, bytes.length-offset)) >= 0) 
	                	        {
	                	            offset += numRead;
	                	        }
	                	    
	                	        
	                	        int ind=352*288*3*frameno;
	                	        
	                	       
	                	       for(int y = 0; y < height; y++){
	                	         	
	                   			for(int x = 0; x < width; x++){
	                	        			byte a = 0;
	                	    				byte r = bytes[ind];
	                	    				byte g = bytes[ind+height*width];
	                	    				byte b = bytes[ind+height*width*2]; 
	                	    				
	                	    				int pix = 0xff000000 | ((r & 0xff) << 16) | ((g & 0xff) << 8) | (b & 0xff);
	                	    				//int pix = ((a << 24) + (r << 16) + (g << 8) + b);
	                	                	img2.setRGB(x, y, pix);
	                	                	ind++;
	                	                }
	                	            }
	                	        	
	                            
	                	        
	                	   
	                			
	                   		
	                		}
	                      
	                            
	                        
	                        } catch (final IOException e2) {
	                           
	                        }
	                		
	                		   
	                		   GridLayout layout = new GridLayout(5,5);
	                           GridBagConstraints c = new GridBagConstraints();
	                      
	                           JButton button2 = new JButton(new ImageIcon(img2));
	                          
	                           button2.setBorder(BorderFactory.createBevelBorder(12));
	                           frame2.setLayout(layout);
	                           frame2.getContentPane().add(button2);
	                           frame2.pack();
	                        //   Thread.sleep (5000);
	                         
	                           frame2.setState ( frame2.ICONIFIED );
	                           frame2.setVisible (false);
	                           // Sleep for 5 seconds, then restore
	                             
	                           frame2.setState ( frame2.NORMAL );

	                           // Sleep for 5 seconds, then kill window
	                         
	                          
	                           frame2.setVisible(true);
	                         //  frame2.setResizable(false); // THEN  resizable = false
	                         //  frame2.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
	                        
	                           button2.addMouseListener(new java.awt.event.MouseAdapter() {
	                               public void mouseEntered(java.awt.event.MouseEvent evt) {
	                               
	                               	button2.setBorder(BorderFactory.createLineBorder(Color.black));
	                               	button2.setBorderPainted(true);
	                               	button.setContentAreaFilled(false);
	                               }
	                               public void mouseExited(java.awt.event.MouseEvent evt) {
	                               	button2.setRolloverEnabled(true);
	                               	button2.setBorderPainted(false);
	                               }
	                           });
	                          
	                           button2.addActionListener(new ActionListener() 
	                           {
	                               public void actionPerformed(ActionEvent e) {
	                            	   BufferedImage img3 = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);

	                            	    try {
	                            	    	if(fileName2.contains("image"))
	                            	    	{
	                            		    File file= new File(dir.getPath()+"/"+fileName2);
	                            		    InputStream is = new FileInputStream(file);

	                            		    long len = file.length();
	                            		    byte[] bytes = new byte[(int)len];
	                            		    
	                            		    int offset = 0;
	                            	        int numRead = 0;
	                            	        while (offset < bytes.length && (numRead=is.read(bytes, offset, bytes.length-offset)) >= 0) {
	                            	            offset += numRead;
	                            	        }
	                            	    

	                            	        
	                            	    		
	                            	    	int ind = 0;
	                            			for(int y = 0; y < height; y++){
	                            		
	                            				for(int x = 0; x < width; x++){
	                            			 
	                            					byte a = 0;
	                            					byte r = bytes[ind];
	                            					byte g = bytes[ind+height*width];
	                            					byte b = bytes[ind+height*width*2]; 
	                            					
	                            					int pix = 0xff000000 | ((r & 0xff) << 16) | ((g & 0xff) << 8) | (b & 0xff);
	                            					//int pix = ((a << 24) + (r << 16) + (g << 8) + b);
	                            					img3.setRGB(x,y,pix);
	                            					ind++;
	                            				}
	                            			}
	                            			
	                            			 // Use a label to display the image
	                                	    JFrame frame3 = new JFrame("Image");
	                                	    JLabel label3 = new JLabel(new ImageIcon(img3));
	                                	    frame3.getContentPane().add(label3, BorderLayout.CENTER);
	                                	    frame3.pack();
	                                	    frame3.setVisible(true);
	                            	    }
	                            	    	else
	                            	    	{
	                            	    		
	                            	    		 //File file = new File("images/"+fileName2);
	                            	    		String file = dir.getPath()+"/"+fileName2;
	                            	    		 BufferedImage[] img4 = readVideoComplete(file);
	                            	    		 displayVideo(img4);
	                            	    	}
	                            	    	
	                            			
	                            	    } catch (FileNotFoundException e3) {
	                            	      e3.printStackTrace();
	                            	    } catch (IOException e3) {
	                            	      e3.printStackTrace();
	                            	    } catch (InterruptedException e1) {
											// TODO Auto-generated catch block
											e1.printStackTrace();
										} 
	                            	    
	                            	    

	                            	   
	                            	   
	                            	   System.out.println(fileName2);
	                	  }
	                });
	                           j++;
	                	}
	                	
	                	/*******************************************************/
	                	
	                	/*******************************************************/
	                	
	                	
	                	
	                }
	            });
	            
	            while(Integer.parseInt(Bhavesh[ptr-1][0])!=(counter)  & counter!=no_of_groups)
	            {
	            	ptr++;
	            	System.out.println("++"+ptr);
	            }
	            System.out.println("ptr"+ptr);
	            counter++;
	            
	    	}
	    	
	    }
	    }
	}




