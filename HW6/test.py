class YoloLikeDetector(nn.Module):             
        """
        The primary purpose of this class is to demonstrate multi-instance object detection with YOLO-like 
        logic.  A key parameter of the logic for YOLO-like detection is the variable 'yolo_interval'.  
        The image gridding that is required is based on the value assigned to this variable.  The grid is 
        represented by an SxS array of cells where S is the image width divided by yolo_interval. So for
        images of size 128x128 and 'yolo_interval=20', you will get a 6x6 grid of cells over the image. Since
        my goal is merely to explain the principles of the YOLO logic, I have not bothered with the bottom
        8 rows and the right-most 8 columns of the image that get left out of the area covered by such a grid.

        An important element of the YOLO logic is defining a set of Anchor Boxes for each cell in the SxS 
        grid.  The anchor boxes are characterized by their aspect ratios.  By aspect ratio I mean the
        'height/width' characterization of the boxes.  My implementation provides for 5 anchor boxes for 
        each cell with the following aspect ratios: 1/5, 1/3, 1/1, 3/1, 5/1.  

        At training time, each instance in the image is assigned to that cell whose central pixel is 
        closest to the center of the bounding box for the instance. After the cell assignment, the 
        instance is assigned to that anchor box whose aspect ratio comes closest to matching the aspect 
        ratio of the instance.

        The assigning of an object instance to a <cell, anchor_box> pair is encoded in the form of a 
        '5+C' element long YOLO vector where C is the number of classes for the object instances.  
        In our cases, C is 3 for the three classes 'Dr_Eval', 'house' and 'watertower', therefore we 
        end up with an 8-element vector encoding when we assign an instance to a <cell, anchor_box> 
        pair.  The last C elements of the encoding vector can be thought as a one-hot representation 
        of the class label for the instance.

        The first five elements of the vector encoding for each anchor box in a cell are set as follows: 
        The first element is set to 1 if an object instance was actually assigned to that anchor box. 
        The next two elements are the (x,y) displacements of the center of the actual bounding box 
        for the object instance vis-a-vis the center of the cell. These two displacements are expressed 
        as a fraction of the width and the height of the cell.  The next two elements of the YOLO vector
        are the actual height and the actual width of the true bounding box for the instance in question 
        as a multiple of the cell dimension.

        The 8-element YOLO vectors are packed into a YOLO tensor of shape (num_cells, num_anch_boxes, 8)
        where num_cell is 36 for a 6x6 gridding of an image, num_anch_boxes is 5.

        Classpath:  RegionProposalGenerator  ->  YoloLikeDetector
        """
        def __init__(self, rpg):
            super(RegionProposalGenerator.YoloLikeDetector, self).__init__()
            self.rpg = rpg
            self.train_dataloader = None
            self.test_dataloader = None

        def show_sample_images_from_dataset(self, rpg):
            data = next(iter(self.train_dataloader))    
            real_batch = data[0]
            first_im = real_batch[0]
            self.rpg.display_tensor_as_image(torchvision.utils.make_grid(real_batch, padding=2, pad_value=1, normalize=True))

        def set_dataloaders(self, train=False, test=False):
            if train:
                dataserver_train = RegionProposalGenerator.PurdueDrEvalMultiDataset(self.rpg, 
                                                       "train", dataroot_train=self.rpg.dataroot_train)
                self.train_dataloader = torch.utils.data.DataLoader(dataserver_train, 
                                                      self.rpg.batch_size, shuffle=True, num_workers=4)
            if test:
                dataserver_test = RegionProposalGenerator.PurdueDrEvalMultiDataset(self.rpg, 
                                                          "test", dataroot_test=self.rpg.dataroot_test)
                self.test_dataloader = torch.utils.data.DataLoader(dataserver_test, 
                                                     self.rpg.batch_size, shuffle=False, num_workers=4)

        def check_dataloader(self, how_many_batches_to_show, train=False, test=False):
            if train:      
                dataloader = self.train_dataloader
            if test:
                dataloader = self.test_dataloader
            for idx, data in enumerate(dataloader): 
                if idx >= how_many_batches_to_show:  
                    break
                im_tensor, seg_mask_tensor, bbox_tensor, bbox_label_tensor, num_objects_in_image = data
                print("\n\nNumber of objects in the batch images: ", num_objects_in_image)
                print("\n\nlabels for the objects found:")
                print(bbox_label_tensor)

                mask_shape = seg_mask_tensor.shape
                logger = logging.getLogger()
                old_level = logger.level
                logger.setLevel(100)
                #  Let's now display the batch images:
                plt.figure(figsize=[15,4])
                plt.imshow(np.transpose(torchvision.utils.make_grid(im_tensor, normalize=True,
                                                                         padding=3, pad_value=255).cpu(), (1,2,0)))
                plt.show()
                #  Let's now display the batch with JUST the masks:
                composite_mask_tensor = torch.zeros(im_tensor.shape[0], 1,128,128)
                for bdx in range(im_tensor.shape[0]):
                    for i in range(num_objects_in_image[bdx]):
                         composite_mask_tensor[bdx] += seg_mask_tensor[bdx][i]
                plt.figure(figsize=[15,4])
                plt.imshow(np.transpose(torchvision.utils.make_grid(composite_mask_tensor, normalize=True,
                                                                         padding=3, pad_value=255).cpu(), (1,2,0)))
                plt.show()
                #  Let's now display the batch and masks in a side-by-side display:
                display_image_and_mask_tensor = torch.zeros(2*im_tensor.shape[0], 3,128,128)
                display_image_and_mask_tensor[:im_tensor.shape[0],:,:,:]  = im_tensor
                display_image_and_mask_tensor[im_tensor.shape[0]:,:,:,:]  = composite_mask_tensor
                plt.figure(figsize=[15,4])
                plt.imshow(np.transpose(torchvision.utils.make_grid(display_image_and_mask_tensor, normalize=False,
                                                                         padding=3, pad_value=255).cpu(), (1,2,0)))
                plt.show()
                #  Let's now display the batch with GT bboxes for the objects:
                im_with_bbox_tensor = torch.clone(im_tensor)
                for bdx in range(im_tensor.shape[0]):
                    bboxes_for_image = bbox_tensor[bdx]
                    for i in range(num_objects_in_image[bdx]):
                        ii = bbox_tensor[bdx][i][0].item()
                        ji = bbox_tensor[bdx][i][1].item()
                        ki = bbox_tensor[bdx][i][2].item()
                        li = bbox_tensor[bdx][i][3].item()
                        im_with_bbox_tensor[bdx,:,ji,ii:ki] = 255    
                        im_with_bbox_tensor[bdx,:,li,ii:ki] = 255                
                        im_with_bbox_tensor[bdx,:,ji:li,ii] = 255  
                        im_with_bbox_tensor[bdx,:,ji:li,ki] = 255  
                plt.figure(figsize=[15,4])
                plt.imshow(np.transpose(torchvision.utils.make_grid(im_with_bbox_tensor, normalize=False,
                                                                         padding=3, pad_value=255).cpu(), (1,2,0)))
                plt.show()
                #  Let's now display the batch with GT bboxes and the object labels
                im_with_bbox_tensor = torch.clone(im_tensor)
                for bdx in range(im_tensor.shape[0]):
                    labels_for_image = bbox_label_tensor[bdx]
                    bboxes_for_image = bbox_tensor[bdx]
                    for i in range(num_objects_in_image[bdx]):
                        ii = bbox_tensor[bdx][i][0].item()
                        ji = bbox_tensor[bdx][i][1].item()
                        ki = bbox_tensor[bdx][i][2].item()
                        li = bbox_tensor[bdx][i][3].item()
                        im_with_bbox_tensor[bdx,:,ji,ii:ki] = 40    
                        im_with_bbox_tensor[bdx,:,li,ii:ki] = 40                
                        im_with_bbox_tensor[bdx,:,ji:li,ii] = 40  
                        im_with_bbox_tensor[bdx,:,ji:li,ki] = 40  
                        im_pil = tvt.ToPILImage()(im_with_bbox_tensor[bdx]).convert('RGBA')
                        text = Image.new('RGBA', im_pil.size, (255,255,255,0))
                        draw = ImageDraw.Draw(text)
                        horiz = ki-10 if ki>10 else ki
                        vert = li
                        label = self.rpg.class_labels[labels_for_image[i]]
                        label = "wtower" if label == "watertower" else label
                        label = "Dr Eval" if label == "Dr_Eval" else label
                        draw.text( (horiz,vert), label, fill=(255,255,255,200) )
                        im_pil = Image.alpha_composite(im_pil, text)
                        im_with_bbox_tensor[bdx] = tvt.ToTensor()(im_pil.convert('RGB'))

                plt.figure(figsize=[15,4])
                plt.imshow(np.transpose(torchvision.utils.make_grid(im_with_bbox_tensor, normalize=False,
                                                                         padding=3, pad_value=255).cpu(), (1,2,0)))
                plt.show()
                logger.setLevel(old_level)




        class AnchorBox( nn.Module ):
            """
            About the role of the 'adx' constructor parameter:  Recall that our goal is to use
            the annotations for each batch to fill up the 'yolo_tensor' that was defined above.
            For case of 5 anchor boxes per cell, this tensor has the following shape:

                     torch.zeros( self.rpg.batch_size, num_yolo_cells, 5, 8 )

            The index 'adx' shown below tells us which of the 5 dimensions on the third axis
            of the 'yolo_tensor' be RESERVED for an anchor box.  We will reserve the 
            coordinate 0 on the third axis for the "1/1" anchor boxes, the coordinate 1 for
            the "1/3" anchor boxes, and so on.  This coordinate choice is set by 'adx'. 
            """
            #               aspect_ratio top_left_corner  anchor_box height & width   anchor_box index
            def __init__(self,   AR,     tlc,             ab_height,   ab_width,      adx):     
                super(RegionProposalGenerator.YoloLikeDetector.AnchorBox, self).__init__()
                self.AR = AR
                self.tlc = tlc
                self.ab_height = ab_height
                self.ab_width = ab_width
                self.adx = adx
            def __str__(self):
                return "AnchorBox type (h/w): %s    tlc for yolo cell: %s    anchor-box height: %d     \
                   anchor-box width: %d   adx: %d" % (self.AR, str(self.tlc), self.ab_height, self.ab_width, self.adx)

    
        def run_code_for_training_multi_instance_detection(self, net, display_labels=False, display_images=False):        
            """
            Version 2.0.6 introduced a loss function that respects the semantics of the different elements 
            of the YOLO vector.  Recall that when you assign an object bounding box to an anchor-box in a 
            specific cell of the grid over the images, you create a 5+C element YOLO vector where C is 
            the number of object classes in your dataset.  Since C=3 in our case, the YOLO vectors in our 
            case are 8-element vectors. See Slide 36 of the Week 8 slides for the meaning to be associated 
            with the different elements of a YOLO vector.

            Lines 64 through 83 in the code shown below are the implementation of the new loss function.

            Since the first element of the YOLO vector is to indicate the presence or the absence of object 
            in a specific anchor-box in a specific cell, I use nn.BCELoss for that purpose.  The next four 
            elements carry purely numerical values that indicate the precise location of the object 
            vis-a-vis the center of the cell to which the object is assigned and also the precise height 
            and the width of the object bounding-box, I use nn.MSELoss for these four elements. The last 
            three elements are a one-hot representation of the object class label, so I use the regular 
            nn.CrossEntropyLoss for these elements.

            As I started writing code for incorporating the nn.CrossEntropyLoss mentioned above, I realized
            that (for purpose of loss calculation) I needed to append one more element to the last three 
            class-label elements of the YOLO vector to take care of the case when there is no object 
            instance present in an anchor box.  You see, the dataset assumes that an image can have a 
            maximum of 5 objects. If an image has fewer than 5 objects, that fact is expressed in the 
            annotations by using the label value of 13 for the 'missing' objects.  To illustrate, say a
            training image has just two objects in it, one being Dr. Eval and the other a house. In this
            case, the annotation for the class labels would be the list [0,1,13,13,13].  If I had not 
            augmented the YOLO vector for loss calculation, the network would be forced to choose
            one of the actual class labels --- 0, 1, or 2 --- in the prediction for a YOLO vector even 
            when there was no object present in the training image for that cell and that anchor box. So 
            when the object label is 13, I throw all the probability mass related to class labels into the 
            additional element (the 9th element) for a YOLO vector.

            See Lines 57 through 60 for the above-mentioned augmentation of the YOLO vectors for all the
            anchor boxes in all of the cells of the grid.

            An important consequence of augmenting the YOLO vectors in the manner explained above is that 
            you must factor the augmentations in the processing of the predictions made by the network.
            An example of that is shown in Line 91 where we supply 9 as the size of the vectors that
            need to be recovered from the predictions.
            """
            if self.rpg.batch_size > 1:                                                                                    ## (1)
                sys.exit("YOLO-like multi-instance object detection has only been tested for batch_size of 1")             ## (2)
            yolo_debug = False
            filename_for_out1 = "performance_numbers_" + str(self.rpg.epochs) + "label.txt"                                
            filename_for_out2 = "performance_numbers_" + str(self.rpg.epochs) + "regres.txt"                               
            FILE1 = open(filename_for_out1, 'w')                                                                           
            FILE2 = open(filename_for_out2, 'w')                                                                           
            net = net.to(self.rpg.device)                                                                                  
            criterion1 = nn.BCELoss()                    # For the first element of the 8 element yolo vector              ## (3)
            criterion2 = nn.MSELoss()                    # For the regression elements (indexed 2,3,4,5) of yolo vector   ## (4)
            criterion3 = nn.CrossEntropyLoss()           # For the last three elements of the 8 element yolo vector        ## (5)
            print("\n\nLearning Rate: ", self.rpg.learning_rate)
            optimizer = optim.SGD(net.parameters(), lr=self.rpg.learning_rate, momentum=self.rpg.momentum)                 ## (6)
            print("\n\nStarting training loop...\n\n")
            start_time = time.perf_counter()
            Loss_tally = []
            elapsed_time = 0.0
            yolo_interval = self.rpg.yolo_interval                                                                         ## (7)
            num_yolo_cells = (self.rpg.image_size[0] // yolo_interval) * (self.rpg.image_size[1] // yolo_interval)         ## (8)
            num_anchor_boxes =  5    # (height/width)   1/5  1/3  1/1  3/1  5/1                                            ## (9)
            max_obj_num  = 5                                                                                               ## (10)
            ## The 8 in the following is the size of the yolo_vector for each anchor-box in a given cell.  The 8 elements 
            ## are: [obj_present, bx, by, bh, bw, c1, c2, c3] where bx and by are the delta diffs between the centers
            ## of the yolo cell and the center of the object bounding box in terms of a unit for the cell width and cell 
            ## height.  bh and bw are the height and the width of object bounding box in terms of the cell height and width.
            for epoch in range(self.rpg.epochs):                                                                           ## (11)
                print("")
                running_loss = 0.0                                                                                         ## (12)
                for iter, data in enumerate(self.train_dataloader):   
                    if yolo_debug:
                        print("\n\n\n======================================= iteration: %d ========================================\n" % iter)
                    yolo_tensor = torch.zeros( self.rpg.batch_size, num_yolo_cells, num_anchor_boxes, 8 )                  ## (13)
                    im_tensor, seg_mask_tensor, bbox_tensor, bbox_label_tensor, num_objects_in_image = data                ## (14)
                    im_tensor   = im_tensor.to(self.rpg.device)                                                            ## (15)
                    seg_mask_tensor = seg_mask_tensor.to(self.rpg.device)                 
                    bbox_tensor = bbox_tensor.to(self.rpg.device)
                    bbox_label_tensor = bbox_label_tensor.to(self.rpg.device)
                    yolo_tensor = yolo_tensor.to(self.rpg.device)
                    if yolo_debug:
                        logger = logging.getLogger()
                        old_level = logger.level
                        logger.setLevel(100)
                        plt.figure(figsize=[15,4])
                        plt.imshow(np.transpose(torchvision.utils.make_grid(im_tensor,normalize=True,padding=3,pad_value=255).cpu(), (1,2,0)))
                        plt.show()
                        logger.setLevel(old_level)
                    cell_height = yolo_interval                                                                            ## (16)
                    cell_width = yolo_interval                                                                             ## (17)
                    if yolo_debug:
                        print("\n\nnum_objects_in_image: ")
                        print(num_objects_in_image)
                    num_cells_image_width = self.rpg.image_size[0] // yolo_interval                                        ## (18)
                    num_cells_image_height = self.rpg.image_size[1] // yolo_interval                                       ## (19)
                    height_center_bb = torch.zeros(im_tensor.shape[0], 1).float().to(self.rpg.device)                      ## (20)
                    width_center_bb = torch.zeros(im_tensor.shape[0], 1).float().to(self.rpg.device)                       ## (21)
                    obj_bb_height = torch.zeros(im_tensor.shape[0], 1).float().to(self.rpg.device)                         ## (22)
                    obj_bb_width =  torch.zeros(im_tensor.shape[0], 1).float().to(self.rpg.device)                         ## (23)

                    ## idx is for object index
                    for idx in range(max_obj_num):                                                                         ## (24)
                        ## In the mask, 1 means good image instance in batch, 0 means bad image instance in batch
#                        batch_mask = torch.ones( self.rpg.batch_size, dtype=torch.int8).to(self.rpg.device)
                        if yolo_debug:
                            print("\n\n               ================  object indexed %d ===============              \n\n" % idx)
                        ## Note that the bounding-box coordinates are in the (x,y) format, with x-positive going to
                        ## right and the y-positive going down. A bbox is specified by (x_min,y_min,x_max,y_max):
                        if yolo_debug:
                            print("\n\nshape of bbox_tensor: ", bbox_tensor.shape)
                            print("\n\nbbox_tensor:")
                            print(bbox_tensor)
                        ## in what follows, the first index (set to 0) is for the batch axis
                        height_center_bb =  (bbox_tensor[0,idx,1] + bbox_tensor[0,idx,3]) // 2                             ## (25)
                        width_center_bb =  (bbox_tensor[0,idx,0] + bbox_tensor[0,idx,2]) // 2                              ## (26)
                        obj_bb_height = bbox_tensor[0,idx,3] -  bbox_tensor[0,idx,1]                                       ## (27)
                        obj_bb_width = bbox_tensor[0,idx,2] - bbox_tensor[0,idx,0]                                         ## (28)
                        if (obj_bb_height < 4.0) or (obj_bb_width < 4.0): continue                                         ## (29)

                        cell_row_indx =  (height_center_bb / yolo_interval).int()          ## for the i coordinate         ## (30)
                        cell_col_indx =  (width_center_bb / yolo_interval).int()           ## for the j coordinates        ## (31)
                        cell_row_indx = torch.clamp(cell_row_indx, max=num_cells_image_height - 1)                         ## (32)
                        cell_col_indx = torch.clamp(cell_col_indx, max=num_cells_image_width - 1)                          ## (33)

                        ## The bh and bw elements in the yolo vector for this object:  bh and bw are measured relative 
                        ## to the size of the grid cell to which the object is assigned.  For example, bh is the 
                        ## height of the bounding-box divided by the actual height of the grid cell.
                        bh  =  obj_bb_height.float() / yolo_interval                                                       ## (34)
                        bw  =  obj_bb_width.float()  / yolo_interval                                                       ## (35)

                        ## You have to be CAREFUL about object center calculation since bounding-box coordinates
                        ## are in (x,y) format --- with x-positive going to the right and y-positive going down.
                        obj_center_x =  (bbox_tensor[0,idx][2].float() +  bbox_tensor[0,idx][0].float()) / 2.0             ## (36)
                        obj_center_y =  (bbox_tensor[0,idx][3].float() +  bbox_tensor[0,idx][1].float()) / 2.0             ## (37)
                        ## Now you need to switch back from (x,y) format to (i,j) format:
                        yolocell_center_i =  cell_row_indx*yolo_interval + float(yolo_interval) / 2.0                      ## (38)
                        yolocell_center_j =  cell_col_indx*yolo_interval + float(yolo_interval) / 2.0                      ## (39)
                        del_x  =  (obj_center_x.float() - yolocell_center_j.float()) / yolo_interval                       ## (40)
                        del_y  =  (obj_center_y.float() - yolocell_center_i.float()) / yolo_interval                       ## (41)
                        class_label_of_object = bbox_label_tensor[0,idx].item()                                            ## (42)
                        ## When batch_size is only 1, it is easy to discard an image that has no known objects in it.
                        ## To generalize this notion to arbitrary batch sizes, you will need a batch mask to indicate
                        ## the images in a batch that should not be considered in the rest of this code.
                        if class_label_of_object == 13: continue                                                           ## (43)
                        AR = obj_bb_height.float() / obj_bb_width.float()                                                  ## (44)
                        if AR <= 0.2:               anch_box_index = 0                                                     ## (45)
                        if 0.2 < AR <= 0.5:         anch_box_index = 1                                                     ## (46)
                        if 0.5 < AR <= 1.5:         anch_box_index = 2                                                     ## (47)
                        if 1.5 < AR <= 4.0:         anch_box_index = 3                                                     ## (48)
                        if AR > 4.0:                anch_box_index = 4                                                     ## (49)
                        yolo_vector = torch.FloatTensor([0,del_x.item(), del_y.item(), bh.item(), bw.item(), 0, 0, 0] )    ## (50)
                        yolo_vector[0] = 1                                                                                 ## (51)
                        yolo_vector[5 + class_label_of_object] = 1                                                         ## (52)
                        yolo_cell_index =  cell_row_indx.item() * num_cells_image_width  +  cell_col_indx.item()           ## (53)
                        yolo_tensor[0,yolo_cell_index, anch_box_index] = yolo_vector                                       ## (54)
                        yolo_tensor_aug = torch.zeros(self.rpg.batch_size, num_yolo_cells, \
                                                                   num_anchor_boxes,9).float().to(self.rpg.device)         ## (55) 
                        yolo_tensor_aug[:,:,:,:-1] =  yolo_tensor                                                          ## (56)
                        if yolo_debug: 
                            print("\n\nyolo_tensor specific: ")
                            print(yolo_tensor[0,18,2])
                            print("\nyolo_tensor_aug_aug: ") 
                            print(yolo_tensor_aug[0,18,2])
                    ## If no object is present, throw all the prob mass into the extra 9th ele of yolo_vector
                    for icx in range(num_yolo_cells):                                                                      ## (57)
                        for iax in range(num_anchor_boxes):                                                                ## (58)
                            if yolo_tensor_aug[0,icx,iax,0] == 0:                                                          ## (59)
                                yolo_tensor_aug[0,icx,iax,-1] = 1                                                          ## (60)
                    if yolo_debug:
                        logger = logging.getLogger()
                        old_level = logger.level
                        logger.setLevel(100)
                        plt.figure(figsize=[15,4])
                        plt.imshow(np.transpose(torchvision.utils.make_grid(im_tensor, normalize=True,
                                                                         padding=3, pad_value=255).cpu(), (1,2,0)))
                        plt.show()

                    optimizer.zero_grad()                                                                                  ## (61)
                    output = net(im_tensor)                                                                                ## (62)
                    predictions_aug = output.view(self.rpg.batch_size,num_yolo_cells,num_anchor_boxes,9)                   ## (63)
                    loss = torch.tensor(0.0, requires_grad=True).float().to(self.rpg.device)                               ## (64)
                    for icx in range(num_yolo_cells):                                                                      ## (65)
                        for iax in range(num_anchor_boxes):                                                                ## (66)
                            pred_yolo_vector = predictions_aug[0,icx,iax]                                                  ## (67)
                            target_yolo_vector = yolo_tensor_aug[0,icx,iax]                                                ## (68)
                            ##  Estimating presence/absence of object and the Binary Cross Entropy section:
                            object_presence = nn.Sigmoid()(torch.unsqueeze(pred_yolo_vector[0], dim=0))                    ## (69)
                            target_for_prediction = torch.unsqueeze(target_yolo_vector[0], dim=0)                          ## (70)
                            bceloss = criterion1(object_presence, target_for_prediction)                                   ## (71)
                            loss += bceloss                                                                                ## (72)
                            ## MSE section for regression params:
                            pred_regression_vec = pred_yolo_vector[1:5]                                                    ## (73)
                            pred_regression_vec = torch.unsqueeze(pred_regression_vec, dim=0)                              ## (74)
                            target_regression_vec = torch.unsqueeze(target_yolo_vector[1:5], dim=0)                        ## (75)
                            regression_loss = criterion2(pred_regression_vec, target_regression_vec)                       ## (76)
                            loss += regression_loss                                                                        ## (77)
                            ##  CrossEntropy section for object class label:
                            probs_vector = pred_yolo_vector[5:]                                                            ## (78)
                            probs_vector = torch.unsqueeze( probs_vector, dim=0 )                                          ## (79)
                            target = torch.argmax(target_yolo_vector[5:])                                                  ## (80)
                            target = torch.unsqueeze( target, dim=0 )                                                      ## (81)
                            class_labeling_loss = criterion3(probs_vector, target)                                         ## (82)
                            loss += class_labeling_loss                                                                    ## (83)
                    if yolo_debug:
                        print("\n\nshape of loss: ", loss.shape)
                        print("\n\nloss: ", loss)
                    loss.backward()                                                                                        ## (84)
                    optimizer.step()                                                                                       ## (85)
                    running_loss += loss.item()                                                                            ## (86)
                    if iter%500==499:                                                                                      ## (87)
                        current_time = time.perf_counter()
                        elapsed_time = current_time - start_time 
                        avg_loss = running_loss / float(1000)                                                              ## (88)
                        print("\n[epoch:%d/%d, iter=%4d  elapsed_time=%5d secs]      mean value for loss: %7.4f" % 
                                                            (epoch+1,self.rpg.epochs, iter+1, elapsed_time, avg_loss))     ## (89)
                        Loss_tally.append(running_loss)
                        FILE1.write("%.3f\n" % avg_loss)
                        FILE1.flush()
                        running_loss = 0.0                                                                                 ## (90)
                        if display_labels:
                            predictions = output.view(self.rpg.batch_size,num_yolo_cells,num_anchor_boxes,9)               ## (91)
                            if yolo_debug:
                                print("\n\nyolo_vector for first image in batch, cell indexed 18, and AB indexed 2: ")
                                print(predictions[0, 18, 2])
                            for ibx in range(predictions.shape[0]):                             # for each batch image     ## (92)
                                icx_2_best_anchor_box = {ic : None for ic in range(36)}                                    ## (93)
                                for icx in range(predictions.shape[1]):                         # for each yolo cell       ## (94)
                                    cell_predi = predictions[ibx,icx]                                                      ## (95)
                                    prev_best = 0                                                                          ## (96)
                                    for anchor_bdx in range(cell_predi.shape[0]):                                          ## (97)
                                        if cell_predi[anchor_bdx][0] > cell_predi[prev_best][0]:                           ## (98)
                                            prev_best = anchor_bdx                                                         ## (99)
                                    best_anchor_box_icx = prev_best                                                        ## (100)
                                    icx_2_best_anchor_box[icx] = best_anchor_box_icx                                       ## (101)
                                sorted_icx_to_box = sorted(icx_2_best_anchor_box,                                   
                                      key=lambda x: predictions[ibx,x,icx_2_best_anchor_box[x]][0].item(), reverse=True)   ## (102)
                                retained_cells = sorted_icx_to_box[:5]                                                     ## (103)
                                objects_detected = []                                                                      ## (104)
                                for icx in retained_cells:                                                                 ## (105)
                                    pred_vec = predictions[ibx,icx, icx_2_best_anchor_box[icx]]                            ## (106)
                                    class_labels_predi  = pred_vec[-4:]                                                    ## (107)
                                    class_labels_probs = torch.nn.Softmax(dim=0)(class_labels_predi)                       ## (108)
                                    class_labels_probs = class_labels_probs[:-1]                                           ## (109)
                                    ##  The threshold of 0.25 applies only to the case of there being 3 classes of objects 
                                    ##  in the dataset.  In the absence of an object, the values in the first three nodes
                                    ##  that represent the classes should all be less than 0.25. In general, for N classes
                                    ##  you would want to set this threshold to 1.0/N
                                    if torch.all(class_labels_probs < 0.25):                                               ## (110)
                                        predicted_class_label = None                                                       ## (111)
                                    else:                                                                                
                                        best_predicted_class_index = (class_labels_probs == class_labels_probs.max())      ## (112)
                                        best_predicted_class_index =torch.nonzero(best_predicted_class_index,as_tuple=True)## (113)
                                        predicted_class_label =self.rpg.class_labels[best_predicted_class_index[0].item()] ## (114)
                                        objects_detected.append(predicted_class_label)                                     ## (115)
                                print("[batch image=%d]  objects found in descending probability order: " % ibx, 
                                                                                                     objects_detected)     ## (116)
                        if display_images:
                            logger = logging.getLogger()
                            old_level = logger.level
                            logger.setLevel(100)
                            plt.figure(figsize=[15,4])
                            plt.imshow(np.transpose(torchvision.utils.make_grid(im_tensor, normalize=True,
                                                                             padding=3, pad_value=255).cpu(), (1,2,0)))
                            plt.show()
                            logger.setLevel(old_level)
            print("\nFinished Training\n")
            plt.figure(figsize=(10,5))
            plt.title("Loss vs. Iterations")
            plt.plot(Loss_tally)
            plt.xlabel("iterations")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig("training_loss.png")
            plt.show()
            torch.save(net.state_dict(), self.rpg.path_saved_yolo_model)
            return net


        def save_yolo_model(self, model):
            '''
            Save the trained yolo model to a disk file
            '''
            torch.save(model.state_dict(), self.rpg.path_saved_yolo_model)


        def run_code_for_testing_multi_instance_detection(self, net, display_images=False):        
            yolo_debug = False
            net.load_state_dict(torch.load(self.rpg.path_saved_yolo_model))
            net = net.to(self.rpg.device)
            yolo_interval = self.rpg.yolo_interval
            num_yolo_cells = (self.rpg.image_size[0] // yolo_interval) * (self.rpg.image_size[1] // yolo_interval)
            num_anchor_boxes =  5    # (height/width)   1/5  1/3  1/1  3/1  5/1
            ##  The next 5 assignment are for the calculations of the confusion matrix:
            confusion_matrix = torch.zeros(3,3)   #  We have only 3 classes:  Dr. Eval, house, and watertower
            class_correct = [0] * len(self.rpg.class_labels)
            class_total = [0] * len(self.rpg.class_labels)
            totals_for_conf_mat = 0
            totals_correct = 0
            ##  We also need to report the IoU values for the different types of objects
            iou_scores = [0] * len(self.rpg.class_labels)
            num_of_validation_images = len(self.test_dataloader)
            print("\n\nNumber of images in the validation dataset: ", num_of_validation_images)
            with torch.no_grad():
                for iter, data in enumerate(self.test_dataloader):
                    ##  In the following, the tensor bbox_label_tensor looks like: tensor([0,0,13,13,13], device='cuda:0',dtype=torch.uint8)
                    ##  where '0' is a genuine class label for 'Dr.Eval' and the number 13 as a label represents the case when there is no
                    ##  object.  You see, each image has a max of 5 objects in it. So the 5 positions in the tensor are for each of those objects.
                    ##  The bounding-boxes for each of those five objects are in the tensor bbox_tensor and segmentation masks in seg_mask_tensor.
                    im_tensor, seg_mask_tensor, bbox_tensor, bbox_label_tensor, num_objects_in_image = data
                    if iter % 50 == 49:
                        if display_images:
                            print("\n\n\n\nShowing output for test image %d: " % (iter+1))
                        im_tensor   = im_tensor.to(self.rpg.device)
                        seg_mask_tensor = seg_mask_tensor.to(self.rpg.device)                 
                        bbox_tensor = bbox_tensor.to(self.rpg.device)
                        output = net(im_tensor)
                        predictions = output.view(self.rpg.batch_size,num_yolo_cells,num_anchor_boxes,9)
                        for ibx in range(predictions.shape[0]):                             # for each batch image
                            ## Our goal is to look through all the cells and identify at most five of the cells/anchor_boxes for 
                            ## the value in the first element of the predicted yolo_vectors is the highest:
                            icx_2_best_anchor_box = {ic : None for ic in range(36)}
                            for icx in range(predictions.shape[1]):                         # for each yolo cell
                                cell_predi = predictions[ibx,icx]               
                                prev_best = 0
                                for anchor_bdx in range(cell_predi.shape[0]):
                                    if cell_predi[anchor_bdx][0] > cell_predi[prev_best][0]:
                                        prev_best = anchor_bdx
                                best_anchor_box_icx = prev_best   
                                icx_2_best_anchor_box[icx] = best_anchor_box_icx
                            sorted_icx_to_box = sorted(icx_2_best_anchor_box, 
                                       key=lambda x: predictions[ibx,x,icx_2_best_anchor_box[x]][0].item(), reverse=True)
                            retained_cells = sorted_icx_to_box[:5]
                        ## We will now identify the objects in the retained cells and also extract their bounding boxes:
                        objects_detected = []
                        predicted_bboxes  = []
                        predicted_labels_for_bboxes = []
                        predicted_label_index_vals = []
                        for icx in retained_cells:
                            pred_vec = predictions[ibx,icx, icx_2_best_anchor_box[icx]]
                            class_labels_predi  = pred_vec[-4:]                        
                            class_labels_probs = torch.nn.Softmax(dim=0)(class_labels_predi)
                            class_labels_probs = class_labels_probs[:-1]
                            if torch.all(class_labels_probs < 0.2): 
                                predicted_class_label = None
                            else:
                                ## Get the predicted class label:
                                best_predicted_class_index = (class_labels_probs == class_labels_probs.max())
                                best_predicted_class_index = torch.nonzero(best_predicted_class_index, as_tuple=True)
                                predicted_label_index_vals.append(best_predicted_class_index[0].item())
                                predicted_class_label = self.rpg.class_labels[best_predicted_class_index[0].item()]
                                predicted_labels_for_bboxes.append(predicted_class_label)
                                ## Analyze the predicted regression elements:
                                pred_regression_vec = pred_vec[1:5].cpu()
                                del_x,del_y = pred_regression_vec[0], pred_regression_vec[1]
                                h,w = pred_regression_vec[2], pred_regression_vec[3]
                                h *= yolo_interval
                                w *= yolo_interval                               
                                cell_row_index =  icx // 6
                                cell_col_index =  icx % 6
                                bb_center_x = cell_col_index * yolo_interval  +  yolo_interval/2  +  del_x * yolo_interval
                                bb_center_y = cell_row_index * yolo_interval  +  yolo_interval/2  +  del_y * yolo_interval
                                bb_top_left_x =  int(bb_center_x - w / 2.0)
                                bb_top_left_y =  int(bb_center_y - h / 2.0)
                                predicted_bboxes.append( [bb_top_left_x, bb_top_left_y, int(w), int(h)] )
                        ## make a deep copy of the predicted_bboxes for eventual visual display:
                        saved_predicted_bboxes =[ predicted_bboxes[i][:] for i in range(len(predicted_bboxes)) ]
                        ##  To account for the batch axis, the bb_tensor is of shape [1,5,4]. At this point, we get rid of the batch axis
                        ##  and turn the tensor into a numpy array.
                        gt_bboxes = torch.squeeze(bbox_tensor).cpu().numpy()
                        ## NOTE:  At this point the GT bboxes are in the (x1,y1,x2,y2) format and the predicted bboxes in the (x,y,h,w) format.
                        ##        where (x1,y1) are the coords of the top-left corner and (x2,y2) of the bottom-right corner of a bbox.
                        ##        The (x,y) for coordinates means x is the horiz coord positive to the right and y is vert coord positive downwards.
                        for pred_bbox in predicted_bboxes:
                            w,h = pred_bbox[2], pred_bbox[3]
                            pred_bbox[2] = pred_bbox[0] + w
                            pred_bbox[3] = pred_bbox[1] + h
                        if yolo_debug:
                            print("\n\nAFTER FIXING:")                        
                            print("\npredicted_bboxes: ")
                            print(predicted_bboxes)
                            print("\nGround Truth bboxes:")
                            print(gt_bboxes)
                        ## These are the mappings from indexes for the predicted bboxes to the indexes for the gt bboxes:
                        mapping_from_pred_to_gt = { i : None for i in range(len(predicted_bboxes))}
                        for i in range(len(predicted_bboxes)):
                            gt_possibles = {k : 0.0 for k in range(5)}      ## 0.0 for IoU 
                            for j in range(len(gt_bboxes)):
                                if all(gt_bboxes[j][x] == 0 for x in range(4)): continue       ## 4 is for the four coords of a bbox
                                gt_possibles[j] = self.IoU_calculator(predicted_bboxes[i], gt_bboxes[j])
                            sorted_gt_possibles =  sorted(gt_possibles, key=lambda x: gt_possibles[x], reverse=True)
                            if display_images:
                                print("For predicted bbox %d: the best gt bbox is: %d" % (i, sorted_gt_possibles[0]))
                            mapping_from_pred_to_gt[i] = (sorted_gt_possibles[0], gt_possibles[sorted_gt_possibles[0]])
                        ##  If you want to see the IoU scores for the overlap between each predicted bbox and all of the individual gt bboxes:
                        if display_images:
                            print("\n\nmapping_from_pred_to_gt: ", mapping_from_pred_to_gt)
                        ## For each predicted bbox, we now know the best gt bbox in terms of the maximal IoU.
                        ## Given a pair of corresponding (pred_bbox, gt_bbox), how do their labels compare is our next question.
                        ## These are the numeric class labels for each of the gt bboxes in the image.
                        gt_labels = torch.squeeze(bbox_label_tensor).cpu().numpy()
                        ## These are the predicted numeric class labels for the predicted bboxes in the image
                        pred_labels_ints = predicted_label_index_vals
                        for i,bbox_pred in enumerate(predicted_bboxes):
                            if display_images:
                                print("for i=%d, the predicted label: %s    the ground_truth label: %s" % (i, predicted_labels_for_bboxes[i], 
                                                                                  self.rpg.class_labels[gt_labels[mapping_from_pred_to_gt[i][0]]]))
                            if gt_labels[pred_labels_ints[i]] != 13:
                                confusion_matrix[gt_labels[mapping_from_pred_to_gt[i][0]]][pred_labels_ints[i]]  +=  1
                            totals_for_conf_mat += 1
                            class_total[gt_labels[mapping_from_pred_to_gt[i][0]]] += 1
                            if gt_labels[mapping_from_pred_to_gt[i][0]] == pred_labels_ints[i]:
                                totals_correct += 1
                                class_correct[gt_labels[mapping_from_pred_to_gt[i][0]]] += 1
                            iou_scores[gt_labels[mapping_from_pred_to_gt[i][0]]] += mapping_from_pred_to_gt[i][1]
                        ## If the user wants to see the image with the predicted bboxes and also the predicted labels:
                        if display_images:
                            predicted_bboxes = saved_predicted_bboxes
                            if yolo_debug:
                                print("[batch image=%d]  objects found in descending probability order: " % ibx, predicted_labels_for_bboxes)
                            logger = logging.getLogger()
                            old_level = logger.level
                            logger.setLevel(100)
                            fig = plt.figure(figsize=[12,12])
                            ax = fig.add_subplot(111)
                            display_scale = 2
                            new_im_tensor = torch.nn.functional.interpolate(im_tensor, scale_factor=display_scale, mode='bilinear', align_corners=False)
                            ax.imshow(np.transpose(torchvision.utils.make_grid(new_im_tensor, normalize=True, padding=3, pad_value=255).cpu(), (1,2,0)))
                            for i,bbox_pred in enumerate(predicted_bboxes):
                                x,y,w,h = np.array(bbox_pred)                                                                     
                                x,y,w,h = [item * display_scale for item in (x,y,w,h)]
                                rect = Rectangle((x,y),w,h,angle=0.0,edgecolor='r',fill = False,lw=2) 
                                ax.add_patch(rect)                                                                      
                                ax.annotate(predicted_labels_for_bboxes[i], (x,y-1), color='red', weight='bold', fontsize=10*display_scale)
                            plt.show()
                            logger.setLevel(old_level)
                                                                                            
            ##  Our next job is to present to the user the information collected for the confusion matrix for the validation dataset:
            if yolo_debug:
                print("\nConfusion Matrix: ", confusion_matrix)
                print("\nclass_correct: ", class_correct)
                print("\nclass_total: ", class_total)
                print("\ntotals_for_conf_mat: ", totals_for_conf_mat)
                print("\ntotals_correct: ", totals_correct)
            for j in range(len(self.rpg.class_labels)):
                print('Prediction accuracy for %5s : %2d %%' % (self.rpg.class_labels[j], 100 * class_correct[j] / class_total[j]))
            print("\n\n\nOverall accuracy of multi-instance detection on %d test images: %d %%" % (num_of_validation_images, 
                                                                                          100 * sum(class_correct) / float(sum(class_total))))
            print("""\nNOTE 1: This accuracy does not factor in the missed detection. This number is related to just the 
       mis-labeling errors for the detected instances.  Percentage of the missed detections are shown in
       the last column of the table shown below.""")
            print("\n\nDisplaying the confusion matrix:\n")
            out_str = "               "
            for j in range(len(self.rpg.class_labels)):  out_str +=  "%15s" % self.rpg.class_labels[j]   
            out_str +=  "%15s" % "missing"
            print(out_str + "\n")
            for i,label in enumerate(self.rpg.class_labels):
                out_percents = [100 * confusion_matrix[i,j] / float(class_total[i]) for j in range(len(self.rpg.class_labels))]
                missing_percent = 100 - sum(out_percents)
                out_percents.append(missing_percent)
                out_percents = ["%.2f" % item.item() for item in out_percents]
                out_str = "%10s:  " % self.rpg.class_labels[i]
                for j in range(len(self.rpg.class_labels)+1): out_str +=  "%15s" % out_percents[j]
                print(out_str)
            print("\n\nNOTE 2: 'missing' means that an object instance of that label was NOT extracted from the image.")   
            print("\nNOTE 3: 'prediction accuracy' means the labeling accuracy for the extracted objects.")   
            print("\nNOTE 4: True labels are in the left-most column and the predicted labels at the top of the table.")

            ##  Finally, we present to the user the IoU scores for each of the object types:
            iou_score_by_label = {self.rpg.class_labels[i] : 0.0 for i in range(len(self.rpg.class_labels))}
            for i,label in enumerate(self.rpg.class_labels):
               iou_score_by_label[self.rpg.class_labels[i]] = iou_scores[i]/float(class_total[i])
            print("\n\nIoU scores for the different types of objects: ")
            for obj_type in iou_score_by_label:
                print("\n    %10s:    %.4f" % (obj_type, iou_score_by_label[obj_type]))



        def IoU_calculator(self, bbox1, bbox2, seg_mask1=None, seg_mask2=None):
            """
            I assume that a bbox is defined by a 4-tuple, with the first two integers standing for the
            top-left coordinate in the (x,y) format and the last two integers for the bottom-right coordinates
            in also the (x,y) format.  By (x,y) format I mean that x stands for the horiz axis with positive to 
            the right and y for the vert coord with positive pointing downwards.  
            """
            union = intersection = 0
            b1x1,b1y1,b1x2,b1y2 = bbox1                             ## b1 refers to bbox1
            b2x1,b2y1,b2x2,b2y2 = bbox2                             ## b2 refers to bbox2
            for x in range(self.rpg.image_size[0]):                 ## image is 128x128
                for y in range(self.rpg.image_size[1]):
                    if  ( ( ( (x >= b1x1) and (x >= b2x1) ) and  ( (y >= b1y1) and (y >= b2y1) ) )  and  \
                        ( ( (x < b1x2) and (x < b2x2) )  and  ((y < b1y2)  and (y < b2y2)) ) ): 
                        intersection += 1
                    if  ( ( (x >= b1x1) and (x <b1x2) ) and  ((y >= b1y1) and (y < b1y2)) ):
                        union += 1            
                    if  ( ( (x >= b2x1) and (x <b2x2) ) and  ((y >= b2y1) and (y < b2y2)) ):
                        union += 1            
            union = union - intersection
            if union == 0.0:
                raise Exception("something_wrong")
            iou = intersection / float(union)
            return iou

