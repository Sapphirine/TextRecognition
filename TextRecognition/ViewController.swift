//
//  ViewController.swift
//  TextRecognition
//
//  Created by fuwenyu on 4/5/17.
//  Copyright © 2017 fuwenyu. All rights reserved.
//

import UIKit
import AWSCore
import AWSS3
import TesseractOCR

class ViewController: UIViewController, UIImagePickerControllerDelegate,
    UINavigationControllerDelegate,
G8TesseractDelegate {
    
    @IBOutlet weak var textView: UITextView!
    @IBOutlet weak var imageView: UIImageView!
    
    var activityIndicator:UIActivityIndicatorView!
    var originalTopMargin:CGFloat!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view, typically from a nib.
        //        if let tesseract = G8Tesseract(language: "eng") {
        //            tesseract.delegate = self
        //            tesseract.image = UIImage(named: "line_truth_330")?.g8_blackAndWhite()
        //            tesseract.recognize()
        //
        //            textView.text = tesseract.recognizedText
        //        }
        
        let tap: UITapGestureRecognizer = UITapGestureRecognizer(target: self, action: "dismissKeyboard")
        view.addGestureRecognizer(tap)
        
    }
    
    //Calls this function when the tap is recognized.
    func dismissKeyboard() {
        //Causes the view (or one of its embedded text fields) to resign the first responder status.
        view.endEditing(true)
    }
    
    
    func progressImageRecognition(for tesseract: G8Tesseract!) {
        print("Recognition Progress \(tesseract.progress) %")
    }
    
    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }
    
    @IBAction func takePhoto(sender: AnyObject) {
        
        view.endEditing(true)
        let imagePickerActionSheet = UIAlertController(title: "Snap/Upload Photo",
                                                       message: nil, preferredStyle: .actionSheet)
        
        // if a camera is avaliable, we add the button to the sheet
        if UIImagePickerController.isSourceTypeAvailable(.camera) {
            let cameraButton = UIAlertAction(title: "Take Photo",
                                             style: .default) { (alert) -> Void in
                                                let imagePicker = UIImagePickerController()
                                                imagePicker.delegate = self
                                                imagePicker.sourceType = .camera
                                                self.present(imagePicker,
                                                             animated: true,
                                                             completion: nil)
            }
            imagePickerActionSheet.addAction(cameraButton)
        }
        
        // add photo button
        let libraryButton = UIAlertAction(title: "Choose Existing",
                                          style: .default) { (alert) -> Void in
                                            let imagePicker = UIImagePickerController()
                                            imagePicker.delegate = self
                                            imagePicker.sourceType = .photoLibrary
                                            self.present(imagePicker,
                                                         animated: true,
                                                         completion: nil)
                                            
        }
        imagePickerActionSheet.addAction(libraryButton)
        
        // cancle button
        let cancelButton = UIAlertAction(title: "Cancel",
                                         style: .cancel) { (alert) -> Void in
        }
        imagePickerActionSheet.addAction(cancelButton)
        
        present(imagePickerActionSheet, animated: true,
                completion: nil)
    }
    
    // scale the input image to a certain size
    func scaleImage(image: UIImage, maxDimension: CGFloat) -> UIImage {
        
        var scaledSize = CGSize(width: maxDimension, height: maxDimension)
        var scaleFactor: CGFloat
        
        if image.size.width > image.size.height {
            scaleFactor = image.size.height / image.size.width
            scaledSize.width = maxDimension
            scaledSize.height = scaledSize.width * scaleFactor
        } else {
            scaleFactor = image.size.width / image.size.height
            scaledSize.height = maxDimension
            scaledSize.width = scaledSize.height * scaleFactor
        }
        
        UIGraphicsBeginImageContext(scaledSize)
        image.draw(in: CGRect(x:0, y:0, width:scaledSize.width, height:scaledSize.height))
        let scaledImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        
        return scaledImage!
    }
    
    // image pick control
    func imagePickerController(_ picker: UIImagePickerController,
                               didFinishPickingMediaWithInfo info: [String : Any]) {
        if let selectedPhoto = info[UIImagePickerControllerEditedImage] as? UIImage{
            //            let scaledImage = scaleImage(image: selectedPhoto, maxDimension: 640)
            imageView.image = selectedPhoto
            
            // fix the display photo size
            imageView.sizeThatFits(selectedPhoto.size)
            
            addActivityIndicator()
            
            dismiss(animated: true, completion: {
                self.performImageRecognition(image: selectedPhoto)
                //                self.performImageRecognition(image: scaledImage)
            })
            
        } else if let selectedPhoto = info[UIImagePickerControllerOriginalImage] as? UIImage{
            //            let scaledImage = scaleImage(image: selectedPhoto, maxDimension: 640)
            imageView.image = selectedPhoto
            
            addActivityIndicator()
            
            dismiss(animated: true, completion: {
                
                self.performImageRecognition(image: selectedPhoto)
                //                self.performImageRecognition(image: scaledImage)
            })
            
        } else {
            imageView.image = nil
        }
    }
    
    func upload_image_to_S3(image: UIImage){
        
        // configure congnito for AWS and iOS
        let credentialsProvider = AWSCognitoCredentialsProvider(regionType:.USEast1,
                                                                identityPoolId:"us-east-1:30d3c939-13f1-411d-a5ae-97e367d9cbb6")
        let configuration = AWSServiceConfiguration(region:.USEast1, credentialsProvider:credentialsProvider)
        
        AWSServiceManager.default().defaultServiceConfiguration = configuration
        
        // upload an image to AWS S3
        let S3_bucket_name = "abd-text-recognition"
        let local_file_name = "CNN_lighter.png"
        let image_url = Bundle.main.url(forResource: "CNN_lighter", withExtension: "png")!
        
        let uploadRequest = AWSS3TransferManagerUploadRequest()
        
        uploadRequest?.body = image_url
        uploadRequest?.key = local_file_name
        uploadRequest?.bucket = S3_bucket_name
        uploadRequest?.contentType = "image/png"
        
        let transferManager = AWSS3TransferManager()
        
        transferManager.upload(uploadRequest!).continueWith(executor: AWSExecutor.mainThread(), block: { (task:AWSTask<AnyObject>) -> Any? in
            
            if let error = task.error as NSError? {
                if error.domain == AWSS3TransferManagerErrorDomain, let code = AWSS3TransferManagerErrorType(rawValue: error.code) {
                    switch code {
                    case .cancelled, .paused:
                        break
                    default:
                        print("Error uploading: \(String(describing: uploadRequest?.key)) Error: \(error)")
                    }
                } else {
                    print("Error uploading: \(String(describing: uploadRequest?.key)) Error: \(error)")
                }
                return nil
            }
            
            let uploadOutput = task.result
            print("Upload complete for: \(String(describing: uploadRequest?.key))")
            return nil
        })
    }
    
    // perform the image recognition locally
    func performImageRecognition(image: UIImage) {
        if let tesseract = G8Tesseract(language: "eng") {
            tesseract.delegate = self
            tesseract.engineMode = .tesseractCubeCombined
            tesseract.pageSegmentationMode = .auto
            tesseract.maximumRecognitionTime = 60.0
            
            tesseract.image = image.g8_blackAndWhite()
            tesseract.recognize()
            
            textView.text = tesseract.recognizedText
        }
        
        // 8
        removeActivityIndicator()
    }
    
    
    func uploadImage(image: UIImage){
        
    }
    
    
    // handle showing and removing the view’s activity indicator:
    func addActivityIndicator() {
        activityIndicator = UIActivityIndicatorView(frame: view.bounds)
        activityIndicator.activityIndicatorViewStyle = .whiteLarge
        activityIndicator.backgroundColor = UIColor(white: 0, alpha: 0.25)
        activityIndicator.startAnimating()
        view.addSubview(activityIndicator)
    }
    
    func removeActivityIndicator() {
        activityIndicator.removeFromSuperview()
        activityIndicator = nil
    }
    
    
    // move the elements of the view in order to prevent the keyboard 
    // from blocking active text fields:
    
    
    
    
}

