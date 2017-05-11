//
//  ViewController.swift
//  TextRecognition
//
//  Created by fuwenyu on 4/5/17.
//  Copyright © 2017 fuwenyu. All rights reserved.
//

import UIKit
import TesseractOCR
import AWSCore
import AWSS3


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
        
        let tap: UITapGestureRecognizer = UITapGestureRecognizer(target: self, action: #selector(ViewController.dismissKeyboard))
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
    
    func size_adjustment(image: UIImage, frame_width: CGFloat) -> CGSize {
        // suppose the width of the fram view is consitant
        let img_width = image.size.width
        let img_height = image.size.height
        
        var result_size = CGSize()
        if img_width > img_height {
            result_size = CGSize(width: frame_width, height: img_height * frame_width / img_width)
            
        } else {
            result_size = CGSize(width: frame_width, height: img_height * frame_width / img_width)
        }
        
        
        return result_size
    }
    
    // image pick control
    func imagePickerController(_ picker: UIImagePickerController,
                               didFinishPickingMediaWithInfo info: [String : Any]) {
        if let selectedPhoto = info[UIImagePickerControllerEditedImage] as? UIImage{
            //            let scaledImage = scaleImage(image: selectedPhoto, maxDimension: 300)
            
            // fix the display photo size
            let origin_width = imageView.frame.width
            imageView.frame.size = size_adjustment(image: selectedPhoto, frame_width: origin_width)
            
            imageView.image = selectedPhoto
            
            addActivityIndicator()
            
            dismiss(animated: true, completion: {
                let image_url = self.upload_image_to_s3(image: selectedPhoto)
                self.get_server_ocr(image_url: image_url)
                //                self.performImageRecognition(image: selectedPhoto) // local OCR
            })
            
        } else if let selectedPhoto = info[UIImagePickerControllerOriginalImage] as? UIImage{
            //            let scaledImage = scaleImage(image: selectedPhoto, maxDimension: 640)
            
            // fix the display photo size
            let origin_width = imageView.frame.width
            imageView.frame.size = size_adjustment(image: selectedPhoto, frame_width: origin_width)
            
            imageView.image = selectedPhoto
            
            addActivityIndicator()
            
            dismiss(animated: true, completion: {
                let image_url = self.upload_image_to_s3(image: selectedPhoto)
                self.get_server_ocr(image_url: image_url)
                //                self.performImageRecognition(image: selectedPhoto) // local OCR
            })
            
        } else {
            imageView.image = nil
        }
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
    
    func upload_image_to_s3(image: UIImage) -> String {
        // Configure AWS Cognito Credentials
        let credentialsProvider = AWSCognitoCredentialsProvider(regionType:.USEast1,
                                                                identityPoolId:"us-east-1:30d3c939-13f1-411d-a5ae-97e367d9cbb6")
        let configuration = AWSServiceConfiguration(region:.USEast1, credentialsProvider:credentialsProvider)
        
        AWSServiceManager.default().defaultServiceConfiguration = configuration
        
        // Set up AWS Transfer Manager Request
        let S3BucketName = "abd-text-recognition"
        let ext = "png"
        let localFileName = "CNN_lighter" // local file name here
        let remoteName = localFileName + "." + ext
        //let fileName = NSUUID().UUIDString + "." + ext
        let imageURL = Bundle.main.url(forResource: localFileName, withExtension: ext)!
        
        let uploadRequest = AWSS3TransferManagerUploadRequest()
        uploadRequest?.body = imageURL
        uploadRequest?.key = remoteName
        uploadRequest?.bucket = S3BucketName
        uploadRequest?.contentType = "image/" + ext
        
        let transferManager = AWSS3TransferManager.default()
        
        // Perform file upload
        transferManager.upload(uploadRequest!)
        transferManager.upload(uploadRequest!).continueWith(executor: AWSExecutor.mainThread(), block: { (task:AWSTask<AnyObject>) -> Any? in
            
            if let error = task.error as NSError? {
                if error.domain == AWSS3TransferManagerErrorDomain, let code = AWSS3TransferManagerErrorType(rawValue: error.code) {
                    switch code {
                    case .cancelled, .paused:
                        break
                    default:
                        print("Error uploading: \(String(describing: uploadRequest!.key)) Error: \(error)")
                    }
                } else {
                    print("Error uploading: \(String(describing: uploadRequest!.key)) Error: \(error)")
                }
                return nil
            }
            
            let uploadOutput = task.result
            print("Upload complete for: \(String(describing: uploadRequest!.key))")
            return nil
        })
        
        print(imageURL)
        
        return "http://s3.amazonaws.com/" + S3BucketName + "/" + remoteName
    }
    
    func get_server_ocr(image_url: String) {
        // create a post request to the server
        var request = URLRequest(url: URL(string: "http://ec2-34-207-139-58.compute-1.amazonaws.com:8888/aply_ocr/")!)
        request.httpMethod = "POST"
        
        var responseString = String("No response...")
        
        request.httpBody = image_url.data(using: .utf8)
        let task = URLSession.shared.dataTask(with: request) { data, response, error in
            guard let data = data, error == nil else {                                                 // check for fundamental networking error
                print("error=\(String(describing: error))")
                return
            }
            
            if let httpStatus = response as? HTTPURLResponse, httpStatus.statusCode != 200 {           // check for http errors
                print("statusCode should be 200, but is \(httpStatus.statusCode)")
                print("response = \(String(describing: response))")
            }
            
            responseString = String(data: data, encoding: .utf8)
            print("responseString = \(String(describing: responseString))")
        }
        task.resume()
        
        textView.text = responseString
    }
    
    
}

