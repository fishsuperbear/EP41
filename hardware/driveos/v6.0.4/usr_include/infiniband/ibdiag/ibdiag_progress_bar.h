/*
 * Copyright (c) 2004-2021 Mellanox Technologies LTD. All rights reserved.
 *
 * This software is available to you under the terms of the
 * OpenIB.org BSD license included below:
 *
 *     Redistribution and use in source and binary forms, with or
 *     without modification, are permitted provided that the following
 *     conditions are met:
 *
 *      - Redistributions of source code must retain the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer.
 *
 *      - Redistributions in binary form must reproduce the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer in the documentation and/or other materials
 *        provided with the distribution.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 */

#ifndef IBDIAG_PROGRESS_BAR_H
#define IBDIAG_PROGRESS_BAR_H

#include "progress_bar.h"

/******************* Progress Bar *******************/

class ProgressBarNodes: public ProgressBar
{
    public:
        ProgressBarNodes(): ProgressBar() { }
        ~ProgressBarNodes() { this->output(); }

        virtual void output();
};

class ProgressBarPorts: public ProgressBar
{
    public:
        ProgressBarPorts(): ProgressBar() { }
        ~ProgressBarPorts() { this->output(); }

        virtual void output();
};

#endif   /* IBDIAG_PROGRESS_BAR_H */
