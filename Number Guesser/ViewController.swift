//
//  ViewController.swift
//  Number Guesser
//
//  Created by Isaac Hunter on 11/17/19.
//  Copyright © 2019 Isaac Hunter. All rights reserved.
//

import UIKit
import CoreGraphics
import Foundation

import Firebase
import FirebaseMLCommon
import FirebaseStorage
import FirebaseFirestore

import Alamofire
import SwiftyJSON


class ViewController: UIViewController, UINavigationControllerDelegate, UIImagePickerControllerDelegate {

    @IBOutlet weak var numberResult: UILabel!
    @IBOutlet weak var imageView: UIImageView!
    @IBOutlet weak var certaintyLabel: UILabel!
    
    @IBAction func predictBtn(_ sender: Any) {
        guard let image = imageView.image?.cgImage
        else { return }
        self.predictImageModel(img: image, interpreter: self.interpreter)
    }
    @IBAction func captureButton(_ sender: Any) {
        self.captureImage()
    }
    
    var imagePicker: UIImagePickerController!
    var interpreter: ModelInterpreter!
    
    let IMG_HEIGHT: NSNumber = 40
    let IMG_WIDTH: NSNumber = 30
    let CHANNELS : NSNumber = 3
    let OUTPUT_DIM : NSNumber = 5
    
    let outputLabels = ["zero", "one", "two", "three", "four"]
    let minCertainty: Double = 0.40
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        let remoteModel = CustomRemoteModel(
            name: "number-classifier"
        )
        
        let downloadConditions = ModelDownloadConditions(
            allowsCellularAccess: true,
            allowsBackgroundDownloading: true
        )
        
        let downloadProgress = ModelManager.modelManager().download(
            remoteModel,
            conditions: downloadConditions
        )
        
        if ModelManager.modelManager().isModelDownloaded(remoteModel) {
            print("Model has downloaded")
            self.interpreter = ModelInterpreter.modelInterpreter(remoteModel: remoteModel)
        } else {
            print("Model failed to download")
        }
        
    }
    
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey: Any]) {
        imagePicker.dismiss(animated: true, completion: nil)
        imageView.image = info[.originalImage] as? UIImage
    }
    
    func captureImage() {
        imagePicker = UIImagePickerController()
        imagePicker.delegate = self
        imagePicker.sourceType = .camera
        
        present(imagePicker, animated: true, completion: nil)
        
    }
    
    
    func predictImageModel(img: CGImage, interpreter: ModelInterpreter) {
        let image = img
        let interpreter = interpreter

        let inputsX = self.preprocessImage(img: image)
        if inputsX != nil {
            print("inputs: ")
            print(inputsX!)
            self.interpretImage(inputs: inputsX!, interpreter: interpreter)
        } else {
            print("interpreter returned nil for inputs")
        }

    }
    
    
    func preprocessImage(img: CGImage) -> ModelInputs? {
//        let val_IMG_SIZE = Int(truncating: IMG_SIZE)
        let val_IMG_HEIGHT = Int(truncating: IMG_HEIGHT)
        let val_IMG_WIDTH = Int(truncating: IMG_WIDTH)
        let image: CGImage = img
        guard let context = CGContext(
          data: nil,
          width: image.width, height: image.height,
          bitsPerComponent: 8, bytesPerRow: image.width * 4,
          space: CGColorSpaceCreateDeviceRGB(),
          bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
        ) else { return nil }

        context.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
        guard let imageData = context.data else { return nil }

        let inputs = ModelInputs()
        var inputData = Data()
        do {
            for row in 0 ..< val_IMG_HEIGHT {
                for col in 0 ..< val_IMG_WIDTH {
                    let offset = 4 * (col * context.width + row)
                    let red = imageData.load(fromByteOffset: offset+1, as: UInt8.self)
                    let green = imageData.load(fromByteOffset: offset+2, as: UInt8.self)
                    let blue = imageData.load(fromByteOffset: offset+3, as: UInt8.self)

                    var normalizedRed = Float32(red) / 255.0
                    var normalizedGreen = Float32(green) / 255.0
                    var normalizedBlue = Float32(blue) / 255.0

                    let elementSize = MemoryLayout.size(ofValue: normalizedRed)
                    var bytes = [UInt8](repeating: 0, count: elementSize)
                    memcpy(&bytes, &normalizedRed, elementSize)
                    inputData.append(&bytes, count: elementSize)
                    memcpy(&bytes, &normalizedGreen, elementSize)
                    inputData.append(&bytes, count: elementSize)
                    memcpy(&bytes, &normalizedBlue, elementSize)
                    inputData.append(&bytes, count: elementSize)
                }
            }
            try inputs.addInput(inputData)
            return inputs
            
        } catch let error {
            print("Failed to add input: \(error)")
            return nil
        }
    }
    
    func interpretImage(inputs: ModelInputs, interpreter: ModelInterpreter) {
        let interpreter: ModelInterpreter = interpreter
        
        let ioOptions = ModelInputOutputOptions()
        do {
            try ioOptions.setInputFormat(index: 0, type: .float32, dimensions: [1, IMG_HEIGHT, IMG_WIDTH, CHANNELS])
            try ioOptions.setOutputFormat(index: 0, type: .float32, dimensions: [1, 5])
        } catch let error as NSError {
            print("Failed to set input or output format with error: \(error.localizedDescription)")
        }
        
        interpreter.run(inputs: inputs, options: ioOptions) { outputs, error in
            guard error == nil, let outputs = outputs else {
                print("interpreter error")
                if (error != nil) {
                    print(error!)
                }
                return
            }
            
            do {
                let result = try outputs.output(index: 0) as! [[NSNumber]]
                let floatArray = result[0].map {
                    a in
                    a.floatValue
                }
                
                self.outputHandler(output: floatArray)
                
            } catch {
                print("something went wrong with the results")
                return
            }
        }
        
    }
    
    func outputHandler(output: [Float]) {
        print("outputs: \(output)")
        
        var outputResult: String!
        var outputCertainty: String!
        var certaintyArray: String! = ""
        
        for i in 0..<(output.count) {
            var varOutput = Double(round(1000 * output[i]) / 1000)
            
            if varOutput >= minCertainty {
                outputResult = self.outputLabels[i]
                outputCertainty = "\(varOutput)"
            }
            else {
                certaintyArray += "[\(i): \(varOutput)]"
            }
        }
        if outputResult != nil {
            self.numberResult.text = outputResult
            self.certaintyLabel.text = "Certainty: \(outputCertainty)"
        }
        else {
            self.numberResult.text = "Uncertain"
            self.certaintyLabel.text = certaintyArray
        }
    }
    
    
    // Connect to the API
    func getOutputData(url: String, parameters: [String: String]) {
        
        Alamofire.request(url, method: .get, parameters: parameters).responseJSON {
            response in
            if response.result.isSuccess {
                print("Success! got the data")
                
                let responseData : JSON = JSON(response.result.value!)
                
                print("JSON response: \(responseData)")
                
            }
            else {
                print("Error \(response.result.error)")
            }
        }
        
    }

}

